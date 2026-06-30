'''
class Poop:
    def __init__(self, name, size):
        self.name = name
        self.size = size

    def getSize(self):
        print("Poops size is: ", self.size)

    def getName(self):
        print("Poops name is: ", self.name)

Kakashka = Poop("Raimbek", 20)
Kakashka.getSize()
Kakashka.getName()
'''

# Simple task

'''
# Your code here:
class Book:
    # 1. Write your __init__ wizard
    def __init__(self,title,author):
        self.title = title
        self.author = author
        
    # 2. Write your get_description method
    def get_description(self):
        return f" {self.title} by {self.author} "

# --- Test Code ---
my_book = Book("The Hobbit", "J.R.R. Tolkien")
print(my_book.get_description()) 
#Expected Output: 'The Hobbit' by J.R.R. Tolkien
'''


#Medium task
'''
# Your code here:
class Character:
    # Write your __init__ and take_damage methods here
    def __init__(self, name, health=100):
        self.name = name
        self.health = health

    def take_damage(self, amount):
        self.health -= amount
        if self.health <= 0:
            print(f"{self.name} has fainted!")
            self.health = 0
        print(f"Players health is: {self.health}")

# --- Test Code ---
hero = Character("Raimbek")
hero.take_damage(30)   # Health becomes 70
hero.take_damage(80)   # Health hits 0 -> Prints "Raimbek has fainted!"
'''

#Hard task
'''
# 1. Standalone function
def check_budget(current_total, item_price, max_budget):
    # Your logic here (returns True or False)
    if current_total + item_price > max_budget:
        return False
    return True

# 2. Class definition
class ShoppingCart:
    # Your __init__ and add_item methods here
    def __init__(self, max_budget, items=[], total_cost=0):
        self.max_budget = max_budget
        self.items = items
        self.total_cost = total_cost

    def add_item(self, item_name, price):
        checking = check_budget(self.total_cost, price, self.max_budget)
        print(self.items)
        if checking:
            self.items.append(item_name)
            self.total_cost += price
            print(f"Added {item_name}")
        else:
            print(f"Declined! {item_name} is too expensive.")

    


# --- Test Code ---
cart = ShoppingCart(50)       # $50 budget
cart.add_item("Pizza", 20)    # Safe -> "Added Pizza"
cart.add_item("Game", 25)     # Safe -> "Added Game" (Total is now $45)
cart.add_item("Soda", 10)     # Exceeds budget! -> "Declined! Soda is too expensive."
'''

#Task on inheritance

class Devices:
    def __init__(self, brand, name):
        self.brand = brand
        self.name = name
        self.is_on = False

    def turn_on(self):
        self.is_on = True
        print(f"{self.brand} {self.name} is now powered ON.")

class SmartSpeaker(Devices):
    def play_music(self, song_title):
        if self.is_on:
            print(f"🔊 Playing '{song_title}' on your {self.brand} speaker.")
        else:
            print("Cannot play music. The device is powered off!")

# --- Test Code ---
my_speaker = SmartSpeaker("Sonos", "Era 100")

my_speaker.play_music("Bohemian Rhapsody") 
# Expected Output: ❌ Cannot play music. The device is powered off!

my_speaker.turn_on() 
# Expected Output: Sonos Era 100 is now powered ON.

my_speaker.play_music("Bohemian Rhapsody") 
# Expected Output: 🔊 Playing 'Bohemian Rhapsody' on your Sonos speaker.