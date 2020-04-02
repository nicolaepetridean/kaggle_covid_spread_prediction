import os
import yaml

# =============================================== CONFIG ===============================================================
class Config:
    def __init__(self, name = None):

        self.experimentName = name
        self.loadStep       = 0
        self.saveDir        = os.path.join('assets', 'checkpoints', self.experimentName)

        self.toDevice = 'cuda:0'  # device on which the trainin will take place

        # training config
        self.noSteps   = 4000    # the number of training steps
        self.batchSize = 64       # the actual size of the batch

        self.winSize  = 14        # window size used for training
        self.predSize = 7         # predict this amount of moments into the future

        self.modelKwargs = {
            'chNo'        : 2,              # number of features(Confirmed and Fatalities)
            'hidChNo'     : 128,            # hidden embedding size
            'future'      : self.predSize,  # predict this amount of moments into the future
            'rnnNoCells'  : 1,              # number of reccurent cell
            'mlpLayerCfg' : [64, 64],       # multi layer perceptron used for processing the RnnCell output
            'mlpActiv'    : 'Tanh',         # mlp activation
            'mlpActivLast': 'Tanh',         # last activation of the mlp layer
        }

        self.optimizerName   = 'RMSprop'      # optimizer name from pytorch library
        self.baseLr          = 1e-6
        self.optimizerKwargs = {          # arguments to be passed to the optimizer
                    'alpha'        : 0.99,
                    'eps'          : 1e-08,
                    'weight_decay' : 5e-4,
                    'momentum'     : 0.9,
                    'centered'     : False
                    }

        # train loss configuration
        self.lossPack  = 'torch.nn'
        self.lossName  = 'MSELoss'
        self.lossKwargs     = {          # argument to be passed to the loss function
                'reduction': 'mean',     # batch reduce method
            }

        # eval loss config
        self.lossEvalPack   = ['core.nn.loss', 'torch.nn']
        self.lossEvalName   = ['RMSLELoss', 'MSELoss']

        # scheduler config
        self.schedPack      = 'core.nn.lr_sched'
        self.schedName      = 'PolyLrSched'
        self.schedKwargs    = {
            'stepSize': 10,
            'iterMax': self.noSteps,
            'powerDecay': 5,
            'warmupIters': 0,
            'warmupFactor': 0.0005,
            'lastEpoch': -1
        }

    # ============================ SAVE ============================================
    def save(self):
        '''
        :param outputPath: the output dir
        :param mode: how to write the file bin or txt
        :param ext: extension used
        :param pickleProt: the pickle protocol
        :return: the save path
        '''

        if not os.path.exists(self.saveDir):
            os.makedirs(self.saveDir)

        assert os.path.exists(self.saveDir), 'Could not create folders'
        saveName = os.path.join(self.saveDir, 'config.yaml')

        with open(saveName, 'w') as f:
            yaml.dump(
                self.__dict__,
                f,
                yaml.Dumper)

        return saveName

    # ============================ LOAD ============================================
    def load(self):
        '''
        :param inputFile: the input file
        :param mode: the mode used for file opening
        :return: the unpickled object
        '''
        saveName = os.path.join(self.saveDir, 'config.yaml')
        with open(saveName, 'w') as f:
            obj = yaml.load(f)

        self.__dict__.update(obj)
        return obj
