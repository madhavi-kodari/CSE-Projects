import random
import string
s=string.ascii_letters+string.digits+string.punctuation
n=int(input("Enter the Length the Password :"))
password="".join(random.choice(s)for i in range(n))
print("Your password is :",password)            
