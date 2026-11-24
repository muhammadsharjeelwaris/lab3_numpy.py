"""CSC350 Lab 3: NumPy hands-on
Name:
Roll number:

Run this file after every task:  python lab3_numpy.py
Commit after every task (git add, git commit -m "Task N done").
"""
import numpy as np

print("NumPy version:", np.__version__)

# =====================================================================
# Task 1: Creating arrays (2 marks)
# =====================================================================
print("\n=== Task 1 ===")
# 1a. Make an array temps with this week's temperatures:
#     31, 28, 35, 30, 33, 29, 36
#     Print it, then print its dtype, shape and size.

# 1b. Make months: the numbers 1 to 12 as a 3 x 4 table
#     (use np.arange and reshape). Print it and its shape.

# 1c. Print an array of five zeros, and a 2 x 3 array of ones.

# 1d. Make a random generator with seed 42 and roll 10 dice
#     (whole numbers from 1 to 6). Print the result.


#solution 1a.
temps= np.array([31, 28, 35, 30, 33, 29, 36])
print(temps, temps.dtype,temps.shape,temps.size)

#solution 1b.
months = np.arange(1,13).reshape(3,4)
print(months,months.shape)

#solution 1c.
print(np.zeros(5))
print(np.ones((2,3)))

#solution 1d.
rng = np.random.default_rng(seed=42)
Rolling_Dice = rng.integers(1,7,size=10)
print(Rolling_Dice)



# =====================================================================
# Task 2: Indexing and masks (2.5 marks)  (use temps from Task 1)
# =====================================================================
print("\n=== Task 2 ===")
# 2a. Print the first and the last temperature.
# 2b. Print the weekend temperatures (the last two days).
# 2c. Print the temperatures in reverse order.
# 2d. Print only the hot days (above 32), how many there are,
#     and their average.  No loops!
# 2e. Print the temperatures from 29 to 33 (both included). Hint: &
# 2f. Use np.where to label each day "Hot" (33 or more) or "OK".


#solution 2a.
print(temps[0],temps[-1])

#solution 2b.
print(temps[-2:])

#solution 2c.
print(temps[::-1])

#solution 2d.
hot_days= temps[temps>32]
a= len(hot_days)
print(hot_days,a,np.mean(hot_days))

#solution 2e.
print(temps[(temps>=29)&(temps<=33)])

#solution 2f.
print(np.where(temps>=33,"Hot","Ok"))


# =====================================================================
# Task 3: Gradebook (3 marks)
# Rows = students, columns = 4 subjects.
# =====================================================================
print("\n=== Task 3 ===")
names = np.array(["Ali", "Sara", "Usman", "Hina", "Bilal"])
marks = np.array([[78, 85, 62, 90],
                  [92, 88, 95, 79],
                  [45, 60, 48, 70],
                  [88, 94, 91, 85],
                  [65, 42, 75, 58]])
# 3a. Print the shape of marks.
# 3b. Print Usman's marks (one row) and all marks of subject 3
#     (column index 2).
# 3c. Print the average of each subject and the average of each student.
# 3d. Print the NAME of the student with the highest average.
#     Hint: argmax gives a position; use it on names.
# 3e. Subject weights are 0.3, 0.3, 0.2, 0.2. Print each student's
#     weighted total, rounded to 1 decimal. (Broadcasting!)
# 3f. Add 5 grace marks to everyone, but no mark may go above 100.
#     Print the new table. Hint: np.clip
# 3g. Print how many students failed (below 50) in each subject.

#solution 3a.
print(marks.shape)

#solution 3b.
print(marks[2],marks[:,2])

#solution 3c.
f=marks.mean(axis=0)
e=marks.mean(axis=1)
print(f,e)

#solution 3d.
print(names[np.argmax(e)])

#solution 3e.
Subject_weights= np.array([0.3, 0.3, 0.2, 0.2])
Total_Weight= marks*Subject_weights
print(np.round(Total_Weight.sum(axis=1),1))

#solution 3f.
g=marks+5
print(np.clip(g,0,100))

#solution 3g.
print(np.sum(marks<50,axis=0))

# =====================================================================
# Task 4: An image is an array (2.5 marks)
# A 5 x 5 grayscale image: 0 = black, 255 = white.
# =====================================================================
print("\n=== Task 4 ===")
img = np.array([[ 10,  10, 200,  10,  10],
                [ 10,  10, 220,  10,  10],
                [210, 230, 250, 230, 210],
                [ 10,  10, 220,  10,  10],
                [ 10,  10, 200,  10,  10]])
# 4a. Print the shape, the darkest value and the brightest value.
# 4b. Print the inverted image (white becomes black). Hint: 255 - ...
# 4c. Print a black-and-white version: 1 where the pixel is above 128,
#     else 0. Hint: a mask, then .astype(int)
# 4d. Print row 2 normalized to the range 0 to 1, rounded to 2 decimals.
# 4e. Print the 3 x 3 centre of the image (crop).



#solution 4a.
print(img.shape,img.max(),img.min())


#solution 4b.
i=255-img
print(i)

#solution 4c.
#this can also be used
#print(np.where(img>128,1,0))
print((img>128).astype(int))

#solution 4d.
r = img[1]/255
print(np.round(r,2))

#solution 4e.
print(img[1:4,1:4])