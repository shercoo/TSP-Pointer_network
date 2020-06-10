#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
int read()
{
    char c;
    int x=0,flag=1;
    while((c=getchar())&&(c<'0'||c>'9'))
        if(c=='-')
            flag=-1;
    do x=x*10+c-'0';
    while((c=getchar())&&c>='0'&&c<='9');
    return x*flag;
}
const int maxn=21;
double dp[1<<maxn][maxn];
double x[maxn],y[maxn];
int pre[1<<maxn][maxn];
int vis[100];
double dist(int a,int b)
{
    return sqrt(pow(x[a]-x[b],2)+pow(y[a]-y[b],2));
}
int main()
{
#ifdef sherco
    // freopen("tmp.in","r",stdin);
    // freopen("tmp.out","w",stdout);
#endif
    // freopen("tsp_5-20_train/tsp_all_len5.txt","r",stdin);
    // freopen("fuck1.out","w",stdout);
    freopen("test_data/tsp40_testdata0.txt","r",stdin);
    freopen("test_data/tsp40_testdata.txt","w",stdout);
    int n=40;
    int p[100];
    int a[100];
    for(int i=1;i<=n;i++)
        a[i]=i;

    for(int C=1;C<=10;C++){
        int flag=0;
        for(int i=0;i<n;i++){
            scanf("%lf%lf",&x[i],&y[i]);
            printf("%.12f %.12f ",x[i],y[i]);
        }
        char ss[10];
        scanf("%s",ss);
        printf("%s ",ss);

        for(int i=0;i<=n;i++)
            scanf("%d",&p[i]);
        double sum=0;
        for(int i=0;i<n;i++){
            sum+=dist(p[i]-1,p[i+1]-1);
        }

        int cur=1;
        double greedy=0,rd=0;
        for(int i=0;i<n-1;i++){
            vis[cur]=C;
            int nx=0;
            double fuck=1e18;
            for(int j=1;j<=n;j++)
                if(vis[j]<C&&dist(cur-1,j-1)<fuck)
                    fuck=dist(cur-1,j-1),nx=j;
            greedy+=dist(cur-1,nx-1);
            cur=nx;
            printf("%d ",cur);
        }
        greedy+=dist(cur-1,0);

        random_shuffle(a+1,a+n+1);
        for(int i=1;i<n;i++){
            rd+=dist(a[i]-1,a[i+1]-1);
        }
        rd+=dist(a[n]-1,a[1]-1);

        printf("%.12f %.12f %.12f\n",sum,greedy,rd);
    }
    // for(int i=0;i<n;i++)
    //     scanf("%lf%lf",&x[i],&y[i]);
    // double a=0,b=0;
    // for(int i=0;i<n;i++)
    //     scanf("%d",&p[i]);
    // for(int i=0;i<n;i++)
    //     a+=sqrt(pow(x[p[i]]-x[p[(i+1)%n]],2)+pow(y[p[i]]-y[p[(i+1)%n]],2));
    // for(int i=0;i<n;i++)
    //     scanf("%d",&p[i]);
    // for(int i=0;i<n;i++)
    //     b+=sqrt(pow(x[p[i]]-x[p[(i+1)%n]],2)+pow(y[p[i]]-y[p[(i+1)%n]],2));
    // printf("%.6f %.6f\n",a,b);
    return 0;
}
