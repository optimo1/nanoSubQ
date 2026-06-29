dict = {"name": "Adil", "age": 16, "city": "Astana"}

print(dict["name"])

dict["name"] = "Raim"
print(dict["name"])

print(dict.get("age"))
print(dict.get("country"))
print(dict.keys())
print(dict.values())
print(dict.items())

print(dict.pop("name"))
print(dict)