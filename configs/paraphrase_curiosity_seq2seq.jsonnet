local embedding_size = 300;
local hidden_size = 100;
local n_dialog_acts = 20;
local num_epochs = 80;
local patience = 3;
local batch_size = 64;
local learning_rate = 0.001;
local num_layers = 1;
local max_decoding_steps = 50;
local sender_size = 10;
local beam_size = 5;
local scheduled_sampling_ratio = 0.0;
local use_bleu = true;
local use_dialog_acts = true;
local filter_user_messages = false;
local filter_empty_facts = false;

{
    train_data_path: 'dialog_data/curiosity_dialogs.train.json',
    validation_data_path: 'dialog_data/curiosity_dialogs.val.json',
    dataset_reader: {
        type: 'curiosity_paraphrase',
        filter_user_messages: filter_user_messages,
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
            values_dim: 64,
            output_projection_dim: hidden_size,
            attention_dropout_prob: 0.1,
        },
        dialog_acts_encoder: {
            input_dim: hidden_size,
            hidden_dims: hidden_size,
            num_layers: 1,
            activations: "relu",
        },
        max_decoding_steps: max_decoding_steps,
        n_dialog_acts: n_dialog_acts,
        beam_size: beam_size,
        target_namespace: "tokens",
        target_embedding_dim: embedding_size,
        scheduled_sampling_ratio: scheduled_sampling_ratio,
        use_bleu: use_bleu,
        use_dialog_acts: use_dialog_acts,
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
        },
        patience: patience,
        //cuda_device: 0,
    },
}
