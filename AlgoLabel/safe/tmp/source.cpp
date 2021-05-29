#include<bits/stdc++.h>
#include <bits/stdc++.h>



#define NM 30002



using namespace std;



int n, l2n, v[NM], c[NM];



int aib[NM];



void upd(int a)

{

    for(int i = a; i <= n; i += i & (-i))

        aib[i]--;

}



int qry(int a)

{

    int ans = 0;

    for(int i = a; i >= 1; i -= i & (-i))

        ans += aib[i];

    return ans;

}



int main()

{

    ifstream fin ("schi.in");

    ofstream fout ("schi.out");

    fin >> n;

    l2n = 1;

    while(l2n * 2 <= n)

        l2n *= 2;

    for(int i = 1; i <= n; i++)

        fin >> v[i];

    for(int i = 1; i <= n; i++)

        aib[i] = i & (-i);

    for(int i = n; i >= 1; i--)

    {

        int l = 1, r = n, mid;

        while(l < r)

        {

            mid = (l + r) / 2;

            if(qry(mid) < v[i])

                l = mid + 1;

            else

                r = mid;

        }

        c[l] = i;

        upd(l);

    }

    for(int i = 1; i <= n; i++)

        fout << c[i] << "\n";

    return 0;

}

