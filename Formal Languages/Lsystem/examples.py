from Lsystem import Lsystem

# I got these example L-systems from https://en.wikipedia.org/wiki/L-system#Examples_of_L-systems.
# Comment out all examples except for the one you wish to see.

dragon_curve = Lsystem(axiom='F', rules={'F': 'F+G', 'G': 'F-G'}, mapping={'G': 'F'})
dragon_curve.render(11, linelength=6)

fractal_plant = Lsystem(axiom='-X', rules={'X': 'F+[[X]-X]-F[-FX]+X', 'F': 'FF'})
fractal_plant.render(6, startingangle=90, linelength=2, turningangle=25)

sierpinski_triangle = Lsystem(axiom='F-G-G', rules={'F': 'F-G+F+G-F', 'G': 'GG'}, mapping={'G': 'F'})
sierpinski_triangle.render(5, linelength=10, turningangle=120)