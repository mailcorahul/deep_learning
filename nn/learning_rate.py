import matplotlib.pyplot as plt

def time_decay(lr=0.01, decay=0.001):
    """
    used in KERAS
    """

    init_lr = lr;
    n_itr = 500;
    LRS = [];
    STEPS = [];
    for itr in range(n_itr):
        lr = lr * (1./(1 + itr*decay));
        LRS.append(lr);
        STEPS.append(itr);

    plt.plot(STEPS, LRS, 'ro');
    plt.axis([0, n_itr, 0, init_lr]);
    plt.show();

if __name__ == '__main__':

    time_decay(lr=1e-5, decay=1e-5);