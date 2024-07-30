
#Similarly incase of designing multiple layer neural network manually, use this concept during gradient descent

def fibonacci_nrml(n):

     if n==1 or n==0:
        return 1
        
     else:
        return  fibonacci_nrml(n-1) + fibonacci_nrml(n-2)  
        
        
def fibonacci(n,d):
   if n in d:
     return d[n]
     
   else:
     d[n]= fibonacci(n-1,d ) + fibonacci(n-2,d)
     return d[n] 

            
d={0:1,1:1}

print(fibonacci(4601025,d))            




