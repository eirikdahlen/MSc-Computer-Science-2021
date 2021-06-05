from matplotlib import pyplot
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_fig(fig_num, fig_name, title, xlabel, ylabel, data, loss=False):
    value = 'loss' if loss else 'accuracy'
    pyplot.style.use('seaborn')
    pyplot.figure(fig_num)
    pyplot.title(title)
    # pyplot.plot(data[f'{value}'], label='train')
    # pyplot.plot(data[f'val_{value}'], label='validation')
    pyplot.plot(data[f'{value}_bertweet'], 'g:', label='train_bertweet', alpha=0.5)
    pyplot.plot(data[f'val_{value}_bertweet'], 'g-', label='validation_bertweet')
    pyplot.plot(data[f'{value}_bert'], ':', color='#069AF3', label='train_bert', alpha=0.5)
    pyplot.plot(data[f'val_{value}_bert'], '-', color='#069AF3', label='validation_bert')
    pyplot.plot(data[f'{value}_distilbert'], 'r:', label='train_distilbert', alpha=0.5)
    pyplot.plot(data[f'val_{value}_distilbert'], 'r-', label='validation_distilbert')
    pyplot.plot(data[f'{value}_ernie'], 'y:', label='train_ernie', alpha=0.5)
    pyplot.plot(data[f'val_{value}_ernie'], 'y-', label='validation_ernie')
    pyplot.plot(data[f'{value}_roberta'], ':', color='sienna', label='train_roberta', alpha=0.5)
    pyplot.plot(data[f'val_{value}_roberta'], '-', color='sienna', label='validation_roberta')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.xticks(np.arange(len(data[f'{value}_bertweet'])), np.arange(1, len(data[f'{value}_bertweet']) + 1))
    pyplot.gca().xaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
    pyplot.legend()
    pyplot.savefig(fig_name)
    pyplot.show()


#
# data = {'loss': [0.36672160029411316, 0.210602805018425, 0.1711597889661789, 0.1337573230266571],
#         'accuracy': [0.874607503414154, 0.9237082600593567, 0.9392663240432739, 0.9522551894187927],
#         'val_loss': [0.21146488189697266, 0.2012200653553009, 0.19919416308403015,
#                               0.18160219490528107],
#         'val_accuracy': [0.922764241695404, 0.9281842708587646, 0.9268292784690857, 0.93360435962677]}
data = {'loss_bertweet': [0.36672160029411316, 0.210602805018425, 0.1711597889661789, 0.1337573230266571],
                 'accuracy_bertweet': [0.874607503414154, 0.9237082600593567, 0.9392663240432739, 0.9522551894187927],
                 'val_loss_bertweet': [0.21146488189697266, 0.2012200653553009, 0.19919416308403015,
                                       0.18160219490528107],
                 'val_accuracy_bertweet': [0.922764241695404, 0.9281842708587646, 0.9268292784690857, 0.93360435962677],
                 'loss_bert': [0.3975338637828827, 0.2423643171787262, 0.1885450780391693, 0.16265597939491272],
                 'accuracy_bert': [0.8625463843345642, 0.9099343419075012, 0.9292749166488647, 0.9390522241592407],
                 'val_loss_bert': [0.26699957251548767, 0.23164451122283936, 0.1867949664592743, 0.2115759551525116],
                 'val_accuracy_bert': [0.9186992049217224, 0.9159891605377197, 0.9295393228530884, 0.925474226474762],
                 'loss_distilbert': [0.37263038754463196, 0.19107936322689056, 0.1449921429157257, 0.10949885100126266],
                 'accuracy_distilbert': [0.8673993945121765, 0.9282044172286987, 0.9481872916221619,
                                         0.9601770043373108],
                 'val_loss_distilbert': [0.2269209623336792, 0.19706836342811584, 0.19694918394088745,
                                         0.2540214955806732],
                 'val_accuracy_distilbert': [0.924119234085083, 0.9281842708587646, 0.9214091897010803,
                                             0.9281842708587646],
                 'loss_ernie': [0.34475526213645935, 0.20348802208900452, 0.1610718071460724, 0.12075725197792053],
                 'accuracy_ernie': [0.8738937973976135, 0.9219954609870911, 0.9414073824882507, 0.955894947052002],
                 'val_loss_ernie': [0.2349449247121811, 0.19755485653877258, 0.18398766219615936, 0.21795931458473206],
                 'val_accuracy_ernie': [0.9119241237640381, 0.9363143444061279, 0.9376693964004517, 0.9281842708587646],
                 'loss_roberta': [0.42734581232070923, 0.2572738528251648, 0.21525336802005768, 0.18701912462711334],
                 'accuracy_roberta': [0.8526263236999512, 0.9032257795333862, 0.9203540086746216, 0.9299172163009644],
                 'val_loss_roberta': [0.2619285583496094, 0.21663667261600494, 0.1970822513103485, 0.2067347764968872],
                 'val_accuracy_roberta': [0.9065040946006775, 0.9281842708587646, 0.9308943152427673,
                                          0.9322493076324463]}

plot_fig(fig_num=1, fig_name='plotter_acc_all.png', title='Accuracy during training and validation for all models',
         xlabel='Epochs', ylabel='Accuracy', data=data)
plot_fig(fig_num=2, fig_name='plotter_loss_all.png', title='Loss during training and validation for all models',
         xlabel='Epochs', ylabel='Loss', data=data, loss=True)
