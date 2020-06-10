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
    freopen("tsp_5-20_train/tsp_all_len10.txt","r",stdin);
    freopen("tsp_5-20_train/tsp_correct_10.txt","w",stdout);
    // freopen("fuck1.out","w",stdout);
    int n=10;
    int p[100];
    for(int C=1;C<=100000;C++){
        for(int i=0;i<n;i++)
            scanf("%lf%lf",&x[i],&y[i]);
        char ss[10];
        scanf("%s",ss);
        for(int i=0;i<=n;i++)
            scanf("%d",&p[i]);
        int U=1<<n;
        for(int s=0;s<U;s++)
            for(int i=0;i<n;i++)
                dp[s][i]=1e10;
        dp[1][0]=0;
        for(int s=0;s<U;s++)
            for(int i=0;i<n;i++)
                if((1<<i)&s)
                {
                    int t=s^(1<<i);
                    for(int j=0;j<n;j++)
                        if((1<<j)&t)
                            if(dp[t][j]+dist(i,j)<dp[s][i])
                            {
                                dp[s][i]=dp[t][j]+dist(i,j);
                                pre[s][i]=j;
                            }
                }
        // for(int i=0;i<n;i++)
        //     printf("%.6f ",dp[U-1][i]);
        // printf("\n");
        // for(int i=0;i<n;i++)
        //     printf("%.6f ",dp[U-1][i]+dist(i,0));
        // printf("\n");
        int r=0,s=U-1;
        double mn=1e10;
        for(int i=1;i<n;i++)
            if(dp[U-1][i]+dist(i,0)<mn)
                r=i,mn=dp[U-1][i]+dist(i,0);
        for(int i=0;i<n;i++)
            printf("%.12f %.12f ",x[i],y[i]);
        printf("output 1 ");
        while(r!=0)
        {
            printf("%d ",r+1);
            int tmp=r;
            r=pre[s][r];
            s^=(1<<tmp);
        }
        printf("1\n");
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
