import turtle
import subprocess

class Lsystem:

    def __init__(self, axiom:str, rules:dict, mapping:dict=None):
        """
        Below are turtle symbols, use *mapping* if you need to map additional symbols to turtle symbols after the final iteration.\n
        **F** → move forward *linelength* pixels\n
        **f** → pick up pen, move forward *linelength* pixels, put down pen\n
        **\\+** → turn left by *turningangle* degrees\n
        **\\-** → turn right by *turningangle* degrees\n
        **\\|** → turn 180 degrees\n
        **[** → push current position and heading to stack\n
        **]** → pick up pen, pop position and heading from stack, put down pen\n
        **\\#** → increase line width by *linewidthincrement*\n
        **!** → decrease line width by *linewidthincrement*\n
        **@** → draw a dot at current position twice as large as line width\n
        **{** → start drawing a polygon with fill\n
        **}** → stop drawing polygon\n
        **>** → upscale *linelength* by *linelengthscalefactor*\n
        **<** → downscale *linelength* by *linelengthscalefactor*\n
        **&** → invert *turningangle*\n
        **)** → increment *turningangle* by *turningangleincrement*\n
        **(** → decrement *turningangle* by *turningangleincrement*
        """
        self.axiom = axiom
        self.rules = rules
        self.mapping = mapping

    def n(self, n:int):
        """
        Generates the *n*-th iteration of the L-system.\n
        """
        string = self.axiom
        for _ in range(n):
            newstring = ''
            for symbol in string:
                if symbol in self.rules:
                    newstring += self.rules[symbol]
                else:
                    newstring += symbol
            string = newstring
        if self.mapping:
            mapped = ''
            for symbol in string:
                if symbol in self.mapping:
                    mapped += self.mapping[symbol]
                else:
                    mapped += symbol
            return mapped
        return string

    def render(self, n:int, animation=False, outputfile:str=None, outputpath:str='', startingpos=(0, 0), startingangle=0, linelength=1, turningangle=90, linewidthincrement=1, linelengthscalefactor=2, turningangleincrement=15):
        """
        Renders the *n*-th iteration of the L-system.\n
        To output a .png, ensure Ghostscript is installed and pass a filename to *outputfile* and a path to *outputpath*.
        """
        screen = turtle.getscreen()
        screen.tracer(animation)
        turtle.hideturtle()
        turtle.speed(0)
        turtle.penup()
        turtle.goto(startingpos)
        turtle.setheading(startingangle)
        turtle.pendown()
        pos = []
        angle = []
        bound = 0
        x0, y0 = startingpos
        for symbol in self.n(n):
            match symbol:
                case 'F':
                    turtle.forward(linelength)
                case 'f':
                    turtle.penup()
                    turtle.forward(linelength)
                    turtle.pendown()
                case '+':
                    turtle.left(turningangle)
                case '-':
                    turtle.right(turningangle)
                case '|':
                    turtle.left(180)
                case '[':
                    pos.append(turtle.pos())
                    angle.append(turtle.heading())
                case ']':
                    turtle.penup()
                    turtle.goto(pos.pop())
                    turtle.setheading(angle.pop())
                    turtle.pendown()
                case '#':
                    turtle.pensize(turtle.pensize()+linewidthincrement)
                case '!':
                    turtle.pensize(turtle.pensize()-linewidthincrement)
                case '@':
                    turtle.dot(2*turtle.pensize())
                case '{':
                    turtle.begin_poly()
                    turtle.begin_fill()
                case '}':
                    turtle.end_poly()
                    turtle.end_fill()
                case '>':
                    linelength *= linelengthscalefactor
                case '<':
                    linelength /= linelengthscalefactor
                case '&':
                    turningangle *= -1
                case ')':
                    turningangle += turningangleincrement
                case '(':
                    turningangle -= turningangleincrement
            if outputfile != None:
                x, y = turtle.pos()
                bound = max(bound, ((x-x0)**2+(y-y0)**2)**0.5)
        screen.update()
        if outputfile != None:
            screen.getcanvas().postscript(file=outputpath+outputfile+'.ps', x=-bound, y=-bound, width=2*bound, height=2*bound)
            subprocess.Popen('gswin64c.exe -sDEVICE=pngalpha -r'+str(int(2*bound))+' -dGraphicsAlphaBits=1 -dFitPage -o '+outputfile+'.png '+outputfile+'.ps', cwd=outputpath)
        turtle.mainloop()
