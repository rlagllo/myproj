input_file = "C:\\Users\\김해창\\Desktop\\airfoil.txt"
output_file = "C:\\Users\\김해창\\Desktop\\airfoilASC.txt"

with open(input_file,'r') as f:
    lines = f.readlines()
    
with open(output_file,'w',encoding='ascii') as f:
    #f.write("NODE,ID,X,Y,Z\n")
    for i,line in enumerate(lines):
        try:
            x, y, z = map(float, line.strip().split('\t'))
            #f.write(f"{i+1}, {x}, {y}, {z}\n")
            f.write(f"{x}, {y}, {z}\n")
        except ValueError as e:
            print(f"Error in line {i+1}: {line.strip()} - {e}")