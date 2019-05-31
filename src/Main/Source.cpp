#include "mainEngine.h"

#define READ_DATA 1

int main(int argc, char *argv[])
{
	
#ifdef READ_DATA
	MainEngine main(1);
	main.ActivateLoadedData();
#endif
#ifndef READ_DATA
	MainEngine main(0);
	main.Activate();
#endif
	return 0;
}
