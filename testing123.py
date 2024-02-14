from time import sleep
filename = "output.txt"

print("Hello")
print("No")

with open(filename, "a") as file:
    for i in range(1,11):
        print(str(i) + "written to file.")

        file.write(str(i) + "\n")

        sleep(10)

print("All numbers printed and written to file!")
