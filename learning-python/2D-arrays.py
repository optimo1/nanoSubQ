#medium task
'''
world_map = [
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 0]
]
count = 0
for row in world_map:
    for seat in row:
        if seat == 1:
            count += 1
print(count)
'''

#hard task

def find_treasure(grid):
    for row in range(len(grid)):
        for col in range(len(grid[row])):
            if grid[row][col] == "Treasure":
                return [row, col]


# Test map for the Hard Task

island_map = [
    ["Sand", "Sand", "Palm Tree"],
    ["Sand", "Treasure", "Rock"],
    ["Palm Tree", "Sand", "Sand"]
]
print(find_treasure(island_map))

# Expected Output from function: [1, 1] (since Treasure is at row 1, column 1)
        
