#
build_dir="build"
mkdir $build_dir

echo "Compiling asm file: $1"
filename=$(echo "$1" | sed -e 's/\.[^.]*$//')

rm -rf $build_dir/$filename.o $build_dir/$filename.out

nasm -f elf64 $1 -o $build_dir/$filename.o

echo "nasm result: $?"
