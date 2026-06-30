#x=10
#while x > 2:
    #print(x)
    #x -= 2


#countdown = 10

# Your code here:
# Loop downward, skip 5, and print "Blastoff!" at the end.

#while countdown > 0:
#    if countdown == 5:
#        pass
#    else: 
#    countdown -= 1
#print("Blastoff!")


#word_count = 0

# Your code here:
# Use while True, ask for input, break on "stop", and track the count.

#while True:
#    user_word = input("Enter a word (or 'stop' to exit): ")
#    if user_word == "stop":
#        break
#    else:
#        word_count += 1
#print(word_count)


balance = 1000
target = 2000
years = 0

# Your code here:
# Loop until balance hits the target. 
# Apply interest, subtract the fee, increase the year count, and print the final years.

while target > balance:
    balance = balance * 1.05
    balance = balance - 25
    years += 1
print(years)
 
