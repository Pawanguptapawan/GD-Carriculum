def solve():
    n=int(input())
    p=list(map(int,input().split()))
    ans=0
    for i in range(n):
        ans+=(p[i]<=i+1)
    return ans

t=int(input())
for _ in range(t):
    print(solve())









