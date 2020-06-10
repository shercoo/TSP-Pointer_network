import matplotlib.pyplot as plt

size=10
step=[]
loss=[]
ans=[]
offset=0

try:
    f=open(str(size)+"result0.txt","r")
    for lines in f:
        data = lines.split("tensor")
        print(data)
        if len(data) > 1:
            s, x, y, a = data[0].strip().split(' ')
            l = data[1].strip().split(' ')[-1]
            step.append(int(s))
            loss.append(float(l))
            ans.append(float(a))
        else:
            s, x, y, a, z, l = data[0].strip().split()
            step.append(int(s))
            loss.append(float(l))
            ans.append(float(a))
    offset=step[-1]
except FileNotFoundError:
    print(size)

f=open(str(size)+"result.txt","r")
for lines in f:
    data=lines.split("tensor")
    print(data)
    if len(data)>1:
        s,x,y,a=data[0].strip().split(' ')
        l=data[1].strip().split(' ')[-1]
        step.append(offset+int(s))
        loss.append(float(l))
        ans.append(float(a))
    else:
        s,x,y,a,z,l=data[0].strip().split()
        step.append(offset+int(s))
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
plt.title('tsp'+str(size)+' loss')
plt.grid()
plt.savefig(str(size)+'loss.png')
plt.clf()
plt.plot(step,ans)
plt.xlabel('step')
plt.ylabel('tour_len')
plt.title('tsp'+str(size)+' tour_len')
plt.grid()
plt.savefig(str(size)+'ans.png')

