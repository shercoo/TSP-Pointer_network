import matplotlib.pyplot as plt

size=10
ss=str(size)
f=open(ss+"result_masked.txt","r")
step=[]
loss=[]
ans=[]
for lines in f:
    data=lines.split("tensor")
    print(data)
    if len(data)>1:
        s,x,y,a=data[0].strip().split(' ')
        l=data[1].strip().split(' ')[-1]
        step.append(int(s))
        loss.append(float(l))
        ans.append(float(a))
    else:
        s,x,a,z,l=data[0].strip().split()
        step.append(int(s))
        loss.append(float(l))
        ans.append(float(a))

print(step)
print(loss)
print(ans)
print(len(loss))
print(len(ans))


plt.plot(step,loss)
plt.xlabel('step')
plt.ylabel('loss')
plt.title('tsp'+ss+' loss (masked)')
plt.grid()
plt.savefig(ss+'loss_masked.png')
plt.clf()
plt.plot(step,ans)
plt.xlabel('step')
plt.ylabel('tour_len')
plt.title('tsp'+ss+' tour_len (masked)')
plt.grid()
plt.savefig(ss+'ans_masked.png')

