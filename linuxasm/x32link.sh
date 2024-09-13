#
build_dir="build"

echo "Linking obj file: $1"
filename=$(echo "$1" | sed -e 's/\.[^.]*$//')

rm -rf $build_dir/$filename.out

ld -m elf_i386 $build_dir/$filename.o -o $build_dir/$filename.out

echo "link result: $?"