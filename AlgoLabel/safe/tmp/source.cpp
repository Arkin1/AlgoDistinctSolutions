#include <bits/stdc++.h>
#include <bits/stdc++.h>

#define NM  2000005

#define MOD int(1e9+9)

///single Hash

using namespace std;

ifstream fin ("strmatch.in");

ofstream fout ("strmatch.out");

int n, m;

int p[NM], h[NM], h2[NM], p2[NM];

long long hs, hs2;

char s[NM], t[NM];

vector<int> rez;

int main()

{

    fin.getline(s, NM);

    fin.getline(t, NM);

    n = strlen(t);

    m = strlen(s);

    p[0] = p2[0] = 1;

    for(int i=1; i<=n; i++)

        p[i] = (1LL*p[i-1]*31)%MOD, p2[i] = (1LL*p2[i-1]*53)%MOD;

    for(int i=0; i<n; i++)

        h[i] = 1LL*(((i>0)? h[i-1]: 0) + 1LL*p[i]*(t[i]-'0'+1) %MOD)%MOD;

    for(int i=0; i<n; i++)

        h2[i] = 1LL*(((i>0)? h2[i-1]: 0) + 1LL*p2[i]*(t[i]-'0'+1) %MOD)%MOD;

    for(int i=0; i<m; i++)

        hs = (1LL*(hs + 1LL*p[i]*(s[i]-'0'+1)%MOD))%MOD;

    for(int i=0; i<m; i++)

        hs2 = (1LL*(hs2 + 1LL*p2[i]*(s[i]-'0'+1)%MOD))%MOD;

    int nr = 0;

    for(int i=m-1; i<n; i++)

        if( ((1LL*hs*p[i-m+1])%MOD) == (1LL*(h[i]- ((i>=m)?h[i-m]:0)+MOD)%MOD) &&

           ((1LL*hs2*p2[i-m+1])%MOD) == (1LL*(h2[i]- ((i>=m)?h2[i-m]:0)+MOD)%MOD))

        {

            if(nr<1000)

                rez.push_back(i-m+1);

            nr++;

        }

    fout << nr << '\n';

    nr = 0;

    for(auto it:rez)

        if(nr > 1000)

            break;

        else fout << it << ' ', nr++;

    return 0;

}

