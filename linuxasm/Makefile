# https://www.youtube.com/watch?v=AG7n0McI20k
# https://www.youtube.com/watch?v=_r7i5X0rXJk
# https://www.youtube.com/watch?v=aMSFaAcup50
# https://www.youtube.com/watch?v=9PeEYcYHHkM
#
# target: dependencies
#	action
#
#

#CFLAGS = -Wno-implicit-function-declaration

# compiler
#CC = nasm

# prefix
PFEF_BUILD = ./build/

all: $(PFEF_BUILD)task6x64.out

$(PFEF_BUILD)task6x64.out: $(PFEF_BUILD)task6x64.o
	ld -m elf_x86_64 $(PFEF_BUILD)task6x64.o -o $(PFEF_BUILD)task6x64.out

# -g debug symbols
# -l listing
$(PFEF_BUILD)task6x64.o: task6x64.asm
	nasm -g -f elf64 task6x64.asm -o $(PFEF_BUILD)task6x64.o -l $(PFEF_BUILD)task6x64.lst


#.DEFAULT_GOAL := task6x64.asm
#.FILE_NAME := task6x64.asm

#all: clean test

#task6x64.asm:
#    nasm -f elf64 task6x64.asm -o "build_dir/task6x64.o"

#ld -m elf_x86_64 .build_dir/task6x64.o -o .build_dir/task6x64.out

clean:
	rm -f $(PFEF_BUILD)task6x64.o
	rm -rf $(PFEF_BUILD)task6x64.out $(PFEF_BUILD)task6x64.lst

#test: task6x64.out

run: $(PFEF_BUILD)task6x64.out
	echo:
	$(PFEF_BUILD)task6x64.out
#    build_dir//task6x64.out

