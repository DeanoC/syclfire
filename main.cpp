#include "al2o3_platform/platform.h"
#include "al2o3_os/file.hpp"
#include "accel_sycl.hpp"
#include "fire.hpp"
#include <curses.h>
#include <stdlib.h>
#include <time.h>

#define DELAYSIZE 200

void myrefresh(void);
void explode(int, int);

short color_table[] =
{
		COLOR_BLACK, COLOR_BLACK, COLOR_RED, COLOR_YELLOW, COLOR_WHITE,
		COLOR_CYAN, COLOR_MAGENTA, COLOR_BLUE,
};

void display(Fire const& world) {

	attrset(A_NORMAL);
	for(uint32_t y = 0; y < world.height;++y) {
		for(uint32_t x = 0;x < world.width;++x) {
			int val = (int)world.hostIntensity[(y * world.width) + x];
			char cv = ' ';
			if(val <= 128) {
				cv = '#';
				val /= 24;
				if(val == 0) {
					mvaddch(y, x, ' ');
				} else {
					attrset(COLOR_PAIR(val));
					mvaddch(y, x, cv);
				}
			} else {
				cv = '$';
				attrset(COLOR_PAIR(6));
				mvaddch(y,x, cv);
			}
		}


	}
	refresh();
}

int main() {
	WINDOW *pdcWindow = initscr();

	if (pdcWindow != nullptr) {
		keypad(stdscr, TRUE);
		nodelay(stdscr, TRUE);
		noecho();

		if (has_colors())
			start_color();
		for (uint16_t i = 0; i < 8; i++) {
			init_pair(i, color_table[i], COLOR_BLACK);
		}
	}
	using namespace Accel;
	Sycl* sycl = Sycl::Create();

	{
		Fire world(128, 32);
		world.init(sycl->getQueue());

		int count = 1;
		bool cont = true;
		while (cont)
		{
			if(pdcWindow == nullptr) {
				cont = (count--) ? true : false;
			} else {
				cont = getch() == ERR;
			}

			cl::sycl::queue& q = sycl->getQueue();

			world.update(q);

			if(pdcWindow != nullptr) {
				world.flushToHost();
				display(world);
			}
		}
	}

	sycl->Destroy();
	if(pdcWindow != nullptr) {
		endwin();
	}

	return 0;
}
