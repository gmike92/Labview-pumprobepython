import pyvisa
rm = pyvisa.ResourceManager()
print("Resources found:")
for res in rm.list_resources():
    print(res)
