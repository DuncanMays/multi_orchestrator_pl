from matplotlib import pyplot as plt

from vary_budget.EOL_data import ideal_EOL, uncertain_EOL

x = [i for i in range(len(ideal_EOL))]
b_tuples = [str(i[0]) for i in ideal_EOL]
a = [i[1] for i in ideal_EOL]
b = [i[1] for i in uncertain_EOL]

plt.plot(x, a, '--', label='ideal_EOL')
plt.plot(x, b, '-', label='uncertain_EOL')
plt.plot(x, [0 for i in ideal_EOL], color='black', label='zero EOL')

plt.title('EOL versus Budget', size=30)
plt.legend(prop={'size': 20})
plt.xlabel('Budgets (mnist_ffn, mnist_cnn)', size=20)
plt.ylabel('Expected Opportunity Loss', size=20)
plt.xticks(x, b_tuples, size=15)

plt.show()