import matplotlib.pyplot as plt
import seaborn as sns
import pickle
plt.rcParams['font.family'] = 'Times New Roman'
def load_var(filename):
    with open(filename,'rb') as f:
        history = pickle.load(f)
    return history
for i in range(6):
    filename = 'history_{}'.format(i)
    history = load_var(filename)
    sns.lineplot(data=history)
handles, labels = plt.gca().get_legend_handles_labels()

# 只显示前两行的图例
plt.legend(handles[:2], labels[:2], loc='lower right')
plt.grid()
plt.savefig('train_history')
plt.show()