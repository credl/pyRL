import curses

class MyConsole:

    stdscr = curses.initscr()

    def __init__(self):
        self.__start_ncurses()

    def __start_ncurses(self):
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(1)

    def end(self):
        curses.nocbreak()
        self.stdscr.keypad(False)
        curses.echo()
        curses.endwin()

    def myprint(self, mystr):
        i = 0
        for line in mystr.split("\n"):
            self.stdscr.addstr(i, 0, line)
            i += 1

    def getch(self):
        return self.stdscr.getch()

    def erase(self):
        self.stdscr.erase()

    def refresh(self):
        self.stdscr.refresh()