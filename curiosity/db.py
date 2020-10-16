#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.

"""
A database containing all the wikipedia/entity linking information.
"""
import random
import re
import subprocess
import os
from contextlib import contextmanager
from collections import defaultdict
from typing import List, Tuple, Dict, NamedTuple
from sqlalchemy import Boolean, Integer, ForeignKey, Column, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import (
    Load,
    sessionmaker,
    scoped_session,
    relationship,
    selectinload,
)
from sqlalchemy.orm.scoping import ScopedSession


def md5sum(filename: str) -> str:
    return (
        subprocess.run(
            f"md5sum {filename}", shell=True, stdout=subprocess.PIPE, check=True
        )
        .stdout.decode("utf-8")
        .split()[0]
    )


def verify_checksum(checksum: str, filename: str) -> None:
    if os.path.exists(filename):
        file_checksum = md5sum(filename)
        if checksum != file_checksum:
            raise ValueError(f"Incorrect checksum for: {filename}")
    else:
        raise ValueError(f"File does not exist: {filename}")


REF_RE = r"< ref >.*?< \/ ref >"


def clean_text(text: str) -> str:
    """
    Sometimes the wiki text has formatting that I missed, like refs.
    To avoid reparsing/relinking text, this is a patch fix to do it JIT
    """
    return re.sub(REF_RE, "", text)


Base = declarative_base()


def create_sql(sql_path: str):
    engine = create_engine(f"sqlite:///{sql_path}")
    Base.metadata.bind = engine
    factory = sessionmaker(bind=engine)
    session_cls = scoped_session(factory)
    return engine, session_cls()


class EntityLink(NamedTuple):
    """
    This represents a single fact in the frontend
    """

    page_entity: str
    mention_entity: str
    section_title: str
    pageviews: int
    context: str
    is_location: bool
    # Having database ids is helpful for backlinking and uniqueness
    fact_id: int
    mention_id: int


class Curriculum(NamedTuple):
    views: int
    entities: List[Tuple[str, int]]


class Fact(Base):
    __tablename__ = "fact"
    id = Column(Integer, primary_key=True)
    page = Column(Text(), nullable=False, index=True)
    section_idx = Column(Integer, nullable=False)
    section_title = Column(Text(), nullable=False, index=True)
    paragraph_idx = Column(Integer, nullable=False)
    text = Column(Text(), nullable=False)
    pageviews = Column(Integer, nullable=False)
    mentions = relationship("Mention")


class Mention(Base):
    __tablename__ = "mention"
    id = Column(Integer, primary_key=True)
    is_location = Column(Boolean, nullable=False)
    pageviews = Column(Integer, nullable=False)
    # Duplicate of Fact.page for speed, these never diverge so its safe
    page = Column(Text(), nullable=False)
    title = Column(Text(), nullable=False)
    fact_id = Column(Integer, ForeignKey("fact.id"), nullable=False)
    fact = relationship("Fact", back_populates="mentions")


class WikiSummary(Base):
    __tablename__ = "wiki"
    id = Column(Integer, primary_key=True)
    title = Column(Text(), nullable=False, index=True)
    text = Column(Text(), nullable=False)
    is_simple = Column(Boolean, nullable=False)


class CuriosityStore:
    """
    Convenience class for reading all data
    """

    def __init__(self, sql_path) -> None:
        self._engine = create_engine(f"sqlite:///{sql_path}")
        Base.metadata.bind = self._engine
        self._pages: List[str] = self._cache_pages()

    @property
    @contextmanager
    def _session_scope(self) -> ScopedSession:
        session = scoped_session(sessionmaker(bind=self._engine))
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _cache_pages(self) -> List[str]:
        with self._session_scope as session:
            rows = session.query(Fact.page).distinct().all()
            return [r[0] for r in rows]

    def get_fact_lookup(self):
        with self._session_scope as session:
            rows = session.query(Fact).all()
            return {f.id: f.text for f in rows}

    def get_focus_entities(self):
        """
        Return all possible focus entities
        """
        return self._pages

    def random_entity(self) -> str:
        if "CURIOSITY_ENTITY" in os.environ:
            entity = os.environ["CURIOSITY_ENTITY"]
            if entity in self._pages:
                return entity
        return random.choice(self._pages)

    def random_sections(self, page_entity: str, n: int) -> List[str]:
        with self._session_scope as session:
            rows = (
                session.query(Fact)
                .filter_by(page=page_entity)
                .filter(Fact.section_title != "Body")
                .group_by(Fact.section_title)
                .all()
            )
            section_names = [r.section_title for r in rows]
            if n > len(section_names):
                raise ValueError(f"Not enough sections: {len(section_names)} vs {n}")
            return random.sample(section_names, n)

    def get_links(self, page_entity: str) -> List[EntityLink]:
        """
        For the given page_entity, return all entity links on the page
        """
        with self._session_scope as session:
            rows = (
                session.query(Fact)
                .filter_by(page=page_entity)
                .options(selectinload(Fact.mentions))
            )
            links = []
            for fact in rows:
                for mention in fact.mentions:
                    links.append(
                        EntityLink(
                            fact.page,
                            mention.title,
                            fact.section_title,
                            mention.pageviews,
                            fact.text,
                            mention.is_location,
                            fact.id,
                            mention.id,
                        )
                    )
            return links

    def get_facts(self, page_entity: str, known_entity: str) -> List[EntityLink]:
        """
        Find all facts on focus_entity's page that match known_entity
        """
        with self._session_scope as session:
            rows = (
                session.query(Mention)
                .filter(Mention.page == page_entity)
                .filter(Mention.title == known_entity)
                .group_by(Mention.fact_id)
                .all()
            )
            return [
                EntityLink(
                    m.fact.page,
                    m.title,
                    m.fact.section_title,
                    m.pageviews,
                    clean_text(m.fact.text),
                    m.is_location,
                    m.fact_id,
                    m.id,
                )
                for m in rows
            ]

    def get_sections(self, page_entity: str) -> List[str]:
        """
        Get all the valid sections for this page
        """
        with self._session_scope as session:
            rows = (
                session.query(Fact)
                .filter_by(page=page_entity)
                .group_by(Fact.section_title)
                .options(Load(Fact).load_only("page", "section_title"))
                .all()
            )
            return [r.section_title for r in rows]

    def get_page_facts(self, page_entity: str) -> List[EntityLink]:
        """
        For the given page_entity, return unique facts
        """
        with self._session_scope as session:
            rows = (
                session.query(Fact)
                .filter_by(page=page_entity)
                .options(selectinload(Fact.mentions))
            )
            links = []
            for fact in rows:
                links.append(
                    EntityLink(
                        fact.page,
                        fact.mentions[0].title,
                        fact.section_title,
                        fact.mentions[0].pageviews,
                        fact.text,
                        fact.mentions[0].is_location,
                        fact.id,
                        fact.mentions[0].id,
                    )
                )
            return links

    def get_section_facts(self, page_entity: str, section: str) -> List[EntityLink]:
        with self._session_scope as session:
            rows = (
                session.query(Fact)
                .filter_by(page=page_entity, section_title=section)
                .options(selectinload(Fact.mentions))
                .all()
            )
            return [
                # Use first mention for now, its not too important, but could be
                # improved to random later
                EntityLink(
                    f.page,
                    f.mentions[0].title,
                    f.section_title,
                    f.mentions[0].pageviews,
                    clean_text(f.text),
                    f.mentions[0].is_location,
                    f.id,
                    f.mentions[0].id,
                )
                for f in rows
            ]

    def get_entity_summary(self, page_entity: str) -> str:
        with self._session_scope as session:
            return (
                session.query(WikiSummary)
                .filter_by(title=page_entity)
                .first()
                .text.strip()
            )

    def prominence_curriculum(self) -> Dict[str, Curriculum]:
        """
        For each page, return a curriculum.
        """
        with self._session_scope as session:
            page_mentions = (
                session.query(Fact)
                .options(
                    selectinload(Fact.mentions),
                    Load(Fact).load_only("page", "pageviews"),
                    Load(Mention).load_only("pageviews", "title"),
                )
                .all()
            )
            curr = defaultdict(lambda: {"views": 0, "entities": set()})
            for fact in page_mentions:
                curr[fact.page]["views"] = fact.pageviews
                current_mentions = curr[fact.page]["entities"]
                for m in fact.mentions:
                    current_mentions.add((m.title, m.pageviews))
            return curr
