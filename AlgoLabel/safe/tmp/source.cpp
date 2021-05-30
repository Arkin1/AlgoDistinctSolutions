#include <bits/stdc++.h>
#include <bits/stdc++.h>

#define Dim 100007

using namespace std;

ifstream f("evaluare.in");

ofstream g("evaluare.out");

int Expresie();

int Termen();

int Factor();

char S[Dim];

int indexxx;



int Expresie()

{

    int r=Termen();

    while(S[indexxx]=='+'||S[indexxx]=='-')

    {

        indexxx++;

        if(S[indexxx-1]=='+') r+=Termen();

        else r-=Termen();

    }

    return r;

}



int Termen()

{

    int r=Factor();

    while(S[indexxx]=='/'||S[indexxx]=='*')

    {

        indexxx++;

        if(S[indexxx-1]=='/') r/=Factor();

        else r*=Factor();

    }

    return r;

}



int Factor()

{

    int sgn=1;

    while(S[indexxx]=='-') sgn=-sgn,indexxx++;



    if(S[indexxx]=='(')

    {

        indexxx++;

        int r=Expresie();

        indexxx++;

        return r*sgn;

    }

    int r=0;



    while(S[indexxx]>='0'&&S[indexxx]<='9')

    r=r*10+(S[indexxx]-'0'),indexxx++;



    return r;

}



int main()

{

    freopen("evaluare.in","r",stdin); freopen("evaluare.out","w",stdout);

        cin.getline(S,Dim);

    g<<Expresie();



    return 0;

}

