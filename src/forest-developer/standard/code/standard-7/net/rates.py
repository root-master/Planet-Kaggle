# learning ate schduler
from net.common import *

# http://elgoacademy.org/anatomy-matplotlib-part-1/
def plot_rates(fig, lrs, num_epoches, title=''):

    epoches = np.arange(0,num_epoches)
    lrs     = lrs[0:num_epoches]

    #get limits
    max_lr  = np.max(lrs)
    xmin=0
    xmax=num_epoches
    dx=2

    ymin=0
    ymax=max_lr*1.2
    dy=(ymax-ymin)/10
    dy=10**math.ceil(math.log10(dy))

    ax = fig.add_subplot(111)
    #ax = fig.gca()
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.set_xticks(np.arange(xmin,xmax+0.0001, dx))
    ax.set_yticks(np.arange(ymin,ymax+0.0001, dy))
    ax.set_xlim(xmin,xmax+0.0001)
    ax.set_ylim(ymin,ymax+0.0001)
    ax.grid(b=True, which='minor', color='black', alpha=0.1, linestyle='dashed')
    ax.grid(b=True, which='major', color='black', alpha=0.4, linestyle='dashed')

    ax.set_xlabel('epoches')
    ax.set_ylabel('learning rate')
    ax.set_title(title)
    ax.plot(epoches, lrs)



## simple stepping rates
class StepLR():
    def __init__(self, rates, steps):
        super(StepLR, self).__init__()
        self.rates = rates
        self.steps = steps

    def get_rate(self, epoch=None, num_epoches=None):

        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                + 'rates=' + str(['%0.4f' % i for i in self.rates]) + '\n' \
                + 'steps=' + str(['%0.0f' % i for i in self.steps]) + ''
        return string


## https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
class DecayLR():
    def __init__(self, base_lr, decay, step):
        super(DecayLR, self).__init__()
        self.step  = step
        self.decay = decay
        self.base_lr = base_lr

    def get_rate(self, epoch=None, num_epoches=None):
        lr = self.base_lr * (self.decay**(epoch // self.step))
        return lr



    def __str__(self):
        string = '(Exp) Decay Learning Rates\n' \
                + 'base_lr=%0.3f, decay=%0.3f, step=%0.3f'%(self.base_lr, self.decay, self.step)
        return string




# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicLR():

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)

    def get_rate(self, epoch=None, num_epoches=None):

        self.trn_iterations += 1
        self.clr_iterations += 1
        lr = self.clr()

        return lr

    def __str__(self):
        string = 'Cyclical Learning Rates\n' \
                + 'base_lr=%0.3f, max_lr=%0.3f'%(self.base_lr, self.max_lr)
        return string








# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    num_epoches=50
    num_its=420


    LR = StepLR(steps=[0,   10,   25,    35,    40,     43], \
                rates=[0.1, 0.01, 0.005, 0.001, 0.0001, -1])

    LR = DecayLR (base_lr=0.1, decay=0.32, step=10)
    LR = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=6., mode='triangular', gamma=1., scale_fn=None, scale_mode='cycle')


    lrs = np.zeros((num_epoches),np.float32)
    for epoch in range(num_epoches):

        lr = LR.get_rate(epoch, num_epoches)
        lrs[epoch] = lr
        if lr<0:
            num_epoches = epoch
            break
        print ('epoch=%02d,  lr=%f'%(epoch,lr))


    #plot
    fig = plt.figure()
    plot_rates(fig, lrs, num_epoches, title=str(LR))
    plt.show()


