#!/usr/bin/env python3
from curiosity.reader import CuriosityDialogReader, USER, ASSISTANT


def test_text_to_instance():
    facts_0 = [
        {'fid': 1, 'used': True},
        {'fid': 1, 'used': False},
        {'fid': 1, 'used': False},
    ]
    facts_1 = [
        {'fid': 1, 'used': False},
        {'fid': 1, 'used': False},
        {'fid': 1, 'used': False},
    ]
    facts_2 = [
        {'fid': 1, 'used': False},
        {'fid': 1, 'used': True},
        {'fid': 1, 'used': True},
    ]
    messages = [
        {'sender': USER, 'message': 'first text', 'liked': False},
        {
            'sender': ASSISTANT,
            'message': 'second text',
            'liked': True,
            'facts': facts_0
        },
        {'sender': USER, 'message': 'third text', 'liked': False},
        {
            'sender': ASSISTANT,
            'message': 'fourth text',
            'liked': True,
            'facts': facts_1
        },
        {'sender': USER, 'message': 'fifth text', 'liked': False},
        {
            'sender': ASSISTANT,
            'message': 'sixth text',
            'liked': False,
            'facts': facts_2
        },
    ]
    dialog = {'messages': messages, 'dialog_id': 0}
    instance = CuriosityDialogReader().text_to_instance(
        dialog, ignore_fact=True
    )
    like_labels = [l.label for l in instance['likes']]
    assert like_labels == [
        'not_liked', 'liked',
        'not_liked', 'liked',
        'not_liked', 'not_liked'
    ]
    fact_labels = instance['fact_labels']
    # Users have 1 dummy fact
    assert len(fact_labels[0].array) == 1
    assert len(fact_labels[2].array) == 1
    assert len(fact_labels[4].array) == 1

    assert fact_labels[0].array[0] == 0
    assert fact_labels[2].array[0] == 0
    assert fact_labels[4].array[0] == 0

    assert list(fact_labels[1].array) == [1, 0, 0]
    assert list(fact_labels[3].array) == [0, 0, 0]
    assert list(fact_labels[5].array) == [0, 1, 1]
