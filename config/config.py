# config/config.py

class config: 
    def __init__(self):

        # よく変更するパラメータ
        self.seed = 42
        self.start_Temp = 1.0
        self.end_Temp = 1.0
        self.schedule = "linear"
        self.name = "Base"



        # あまり変更しないパラメータ
        self.source = "en"
        self.target = "de"
        self.ts = "de-en"
        self.wmt = "wmt14"
        self.vocab_size = 32000
        self.train_limit_data = 300 #Noneと書いてはならない(max_train_stepsの計算に必要)
        self.val_limit_data = 10
        self.max_len = 128
        self.SPM_MODEL_PATH = f"saved_tokenizer_data/spm{self.vocab_size}_{self.source}{self.target}.model"
        self.d_model = 512
        self.n_head = 8
        self.ffn_hidden = 2048
        self.n_layers = 6
        self.drop_prob = 0.1
        self.batch_size = 64
        self.epochs = 5
        
        self.use_noam = True
        self.warmup_steps = 14060
        self.fixed_lr = 1e-4
        self.use_wandb = False
        
        self.dump_attention = True