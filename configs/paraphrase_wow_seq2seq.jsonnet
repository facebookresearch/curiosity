local embedding_size = 300;
local hidden_size = 128;
local da_hidden_size = 16;
local n_dialog_acts = 20;
local num_epochs = 100;
local patience = 10;
local batch_size = 64;
local learning_rate = 0.01;
local num_layers = 1;
local max_decoding_steps = 50;
local sender_size = 10;
local beam_size = 5;
local scheduled_sampling_ratio = 0.1;
local use_bleu = true;
local use_dialog_acts = true;
local filter_empty_facts = true;

{
    train_data_path: 'dialog_data/wow_dialogs.train.tsv',
    validation_data_path: 'dialog_data/wow_dialogs.val.tsv',
    dataset_reader: {
        type: 'fact_paraphrase',
        filter_empty_facts: filter_empty_facts,
    },
    model: {
        type: 'curiosity_paraphrase_seq2seq',
        source_embedder: {
            token_embedders: {
                tokens: {
                    type: 'embedding',
                    embedding_dim: embedding_size,
                    pretrained_file: "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.300d.txt.gz",
                    trainable: false,
                },
            },
        },
        source_encoder: {
            type: 'multi_head_self_attention',
            num_heads: 4,
            input_dim: embedding_size,
            attention_dim: hidden_size,
            values_dim: hidden_size,
            output_projection_dim: hidden_size,
            attention_dropout_prob: 0.3,
        },
        dialog_acts_encoder: {
            input_dim: da_hidden_size,
            hidden_dims: da_hidden_size,
            num_layers: 1,
            activations: "linear",
        },
        max_decoding_steps: max_decoding_steps,
        n_dialog_acts: n_dialog_acts,
        beam_size: beam_size,
        target_namespace: "tokens",
        target_embedding_dim: embedding_size,
        scheduled_sampling_ratio: scheduled_sampling_ratio,
        use_bleu: use_bleu,
        use_dialog_acts: use_dialog_acts,
        regularizers: [
            ['_output_projection', {'type': 'l2', 'alpha': 0.01}]
        ]
    },
    iterator: {
        type: 'basic',
        batch_size: batch_size,
    },
    trainer: {
        num_epochs: num_epochs,
        optimizer: {
            type: 'adam',
            lr: learning_rate,
            betas: [0.9, 0.98],
            eps: 1e-08
        },
        patience: patience,
        //cuda_device: 0,
    },
}
