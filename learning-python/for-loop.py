#prices = [10, 15, 8, 22, 14, 5]
#budget = 50

# Your code here: 
# Total up the prices, stop if you cross $50, and print the final total.
#sum = 0
#for i in range(len(prices)):
#    if sum + prices[i] <= budget:
#        sum += prices[i]
#    else:
#       break
#print(sum)

#emails = ["alice@email.com", "bob_at_email.com", "charlie@email.com", "david_no_at", "eve@email.com"]

# Your code here:
# Loop using index positions, filter out the bad emails, and print the seats.
#for item in range(len(emails)):
#    if "@" in emails[item]:
#        print(f"Seat {item}: {emails[item]}") 
#    else:
#        continue


weekly_prices = [120, 95, 210, 185, 310, 140, 290, 85, 400]

# Your code here:
# Isolate the range from index 2 to 6, find the max and min within that section.
highest_item = weekly_prices[2]
lowest_item = weekly_prices[2]
for item in weekly_prices[2:7]:
    if highest_item < item:
        highest_item = item
    if lowest_item > item:
        lowest_item = item
print(highest_item, lowest_item)

# or just 
# mid_week = weekly_prices[2:7]
# print(max(mid_week), min(mid_week))
