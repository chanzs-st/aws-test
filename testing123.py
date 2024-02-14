from time import sleep
filename = "output.txt"

print("Hello")
print("No")
for i in range(1,101):
    with open(filename, "a") as file:
        print(str(i) + " written to file.")

        file.write(str(i) + "\n")

    sleep(10)

print("All numbers printed and written to file!")
