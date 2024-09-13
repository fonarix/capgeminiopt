#
build_dir="build"

echo "Linking obj file: $1"
filename=$(echo "$1" | sed -e 's/\.[^.]*$//')

rm -rf $build_dir/$filename.out

ld -m elf_x86_64 $build_dir/$filename.o -o $build_dir/$filename.out

echo "link result: $?"