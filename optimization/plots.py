import numpy as np
import matplotlib.pyplot as plt

# Plot Total number of messages exhanged


nop = np.array([100, 200, 300, 400]) # The last element is actually 500, I changed it for the sake of the plot
dons = np.array([676071, 3237133, 5871463, 16380174])
dons_log = np.log10(dons)
dons_nav = np.array([2, 2, 2, 5])

desh = np.array([5400, 10800, 16200, 27000])
desh_log = np.log10(desh)

bitcoin = np.array([20000000, 40000000, 60000000, 100000000])
bitcoin_log = np.log10(bitcoin)

mine = np.array([522733, 1193253, 2011761, 3352702])
mine_log = np.log10(mine)
mine_nav = np.array([7, 8, 9, 9])

mine2 = np.array([153022, 306259, 459305, 1874468])



#fig, ax = plt.subplots()
fig = plt.figure(figsize=(20, 10), dpi=200)
plt.rcParams["font.size"] = 25
plt.bar(nop + 00.0, dons, 20)
#plt.bar(nop + 20.0, desh_log, 20)
plt.bar(nop + 20.0, mine2, 20)
# plt.bar(nop + 40.0, bitcoin, 20)
plt.xlabel("Number of Peers")
plt.ylabel("Total Number of Control Messages")
plt.title("Number of Control Messages Exchanged")
plt.legend(('Baniata et al.', 'AOPTABI'))

# plt.xticks(nop, [100, 200, 300, 500])
# plt.twinx()
# plt.plot(nop, dons_nav, marker="o", linewidth=3, color="green")
# plt.plot(nop, mine_nav, marker="o", linewidth=3, color="red")
# plt.legend(('Baniata et al.', 'AOPTABI'))

plt.savefig("control_messages.jpg")
plt.savefig("control_messages.pdf")
plt.show()




nop = np.array([100, 200, 300, 400]) # The last element is actually 500, I changed it for the sake of the plot
dons = np.array([15.9, 25.84, 33.6, 55.4])
dons_log = np.log10(dons)

# desh = np.array([5400, 10800, 16200, 27000])
# desh_log = np.log10(desh)
# bitcoin = np.array([20000000, 40000000, 60000000, 100000000])
# bitcoin_log = np.log10(bitcoin)

mine1 = np.array([163.17640484196636, 362.5259207171184, 556.207044682009, 1094.267706456063])
mine1_log = np.log10(mine1)

mine2 = np.array([76.20281571800192, 227.06797571300467, 326.23622864001675, 505.6409465689933])
mine2_log = np.log10(mine2)

mine3 = np.array([61.12172541800038, 124.80041896000682, 212.11329104498998, 373.86322064700744])
mine3_log = np.log10(mine3)

# mine_av = np.array([a_100_av, a_200_av, a_300_av,a_500_av])
# mine_av_log = np.log10(mine_av)


# print(mine_av)
# print(mine_av_log)



# fig, ax = plt.subplots()
fig = plt.figure(figsize=(20, 10), dpi=200)
plt.rcParams["font.size"] = 25
plt.bar(nop + 00.0, dons, 20)
# plt.bar(nop + 20.0, desh_log, 20)
plt.bar(nop + 20.0, mine3, 20)
# plt.bar(nop + 40.0, mine3, 20)
plt.xlabel("Number of Peers")
plt.ylabel("Time (seconds)")
plt.title("Graph Generation Time")
plt.legend(('Baniata et al.', 'Aoptabi',))
plt.xticks(nop, [100, 200, 300, 500])

plt.savefig("runtime.pdf")
plt.savefig("runtime.jpg")
plt.show()