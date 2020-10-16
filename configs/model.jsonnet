local embedding_size = 300;
local hidden_size = 100;
local num_epochs = 20;
local patience = 3;
local glove_batch_size = 64;
local bert_batch_size = 4;
local learning_rate = 0.001;
local num_layers = 1;
local sender_size = 10;
local act_size = 10;
local mention_size = 100;

local bert_name = '/tmp/curiosity_bert_lm';
local bert_indexer = {
    bert: {
        type: "bert-pretrained",
        pretrained_model: bert_name
    },
};
local bert_tokenizer = {
    type: 'pretrained_transformer',
    model_name: bert_name,
    # This should match uncased/cased
    do_lowercase: true,
};

local glove_indexer = {
    tokens: {
        type: "single_id",
        lowercase_tokens: true
    }
};
local glove_tokenizer = {
    type: 'word'
};
local glove_embedder = {
    # TODO: add character embeddings, but this is not trivial due to
    # double time distributed nature
    token_embedders: {
        tokens: {
            type: 'embedding',
            embedding_dim: embedding_size,
            pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
            trainable: true,
        },
    },
};
local glove_context = {
    type: 'lstm',
    input_size: embedding_size,
    hidden_size: hidden_size,
    bidirectional: true,
};

local glove_dim = hidden_size;
local bert_dim = 768;

// use_glove and use_bert must be xor
function(use_glove=true, use_bert=false,
         disable_known_entities=false,
         disable_dialog_acts=false,
         disable_likes=false,
         disable_facts=false) {
    train_data_path: 'dialog_data/curiosity_dialogs.train.json',
    validation_data_path: 'dialog_data/curiosity_dialogs.val.json',
    dataset_reader: {
        type: 'curiosity_dialog',
        token_indexers: if use_glove then glove_indexer else bert_indexer,
        tokenizer: if use_glove then glove_tokenizer else bert_tokenizer,
    },
    model: {
        type: 'curiosity_model',
        use_bert: use_bert,
        use_glove: use_glove,
        bert_name: bert_name,
        bert_trainable: false,
        mention_embedder: {
            mentions: {
                type: 'embedding',
                embedding_dim: mention_size,
                trainable: false,
                # wget http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2
                # bzip2 -d enwiki_20180420_100d.txt.bz2
                # ./cli filter-emb enwiki_20180420_100d.txt wiki2vec_entity_100d.txt
                pretrained_file: "dialog_data/wiki2vec_entity_100d.txt",
            },
        },
        utter_embedder: if use_glove then glove_embedder else null,
        utter_context: if use_glove then glove_context else null,
        dialog_context: {
            // TODO: GRU?
            type: 'lstm',
            // forward/backward states, sende + act, known + focus
            input_size: (if use_glove then 2 * glove_dim else bert_dim) + sender_size + act_size + 2 * mention_size,
            hidden_size: hidden_size,
            // TODO: can be bidirectional, but need to take care
            // not to leak info backwards
            bidirectional: false,
        },
        fact_ranker: {
            type: "mean_logit_ranker",
            similarity_function: {
                type: "bilinear",
                tensor_1_dim: hidden_size,
                tensor_2_dim: (if use_glove then 2 * glove_dim else bert_dim),
            }
        },
        dropout_prob: 0.5,
        sender_emb_size: sender_size,
        act_emb_size: act_size,
        disable_known_entities: disable_known_entities,
        disable_dialog_acts: disable_dialog_acts,
        disable_likes: disable_likes,
        disable_facts: disable_facts,
        # Weight on the fact loss overall
        fact_loss_weight: 3.0,
        # Weight on fact positive class
        fact_pos_weight: 9.0,
    },
    iterator: {
        type: 'basic',
        batch_size: if use_glove then glove_batch_size else bert_batch_size,
    },
    trainer: {
        num_epochs: num_epochs,
        optimizer: {
            type: 'adam',
            lr: learning_rate,
        },
        "validation_metric": "+total",
        patience: patience,
        num_serialized_models_to_keep: 5,
        //cuda_device: 0,
    },
}
