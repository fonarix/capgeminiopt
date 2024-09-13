;
; ./x64compile.sh task1x64.asm
; ./x64link.sh task1x64.o
; ./build/task1x64.out
; get result call: echo $?
;
; links:
; https://stackoverflow.com/questions/60095718/linking-a-compiled-assembly-and-c-file-with-ld
; https://en.wikibooks.org/wiki/X86_Assembly/Interfacing_with_Linux


%include "sys_codes.inc"
%include "unistd_64h.inc"


global _start               ;must be declared for linker (gcc)

section .data
    val1        dd 1
    val2        dd 7
    val3        dd 9
    outresult   dd 0xFFFFFFFF

; Code goes in text section
section .text

_start:
    mov rax, [val1]
    add rax, [val2]
    add rax, [val3]
    mov [outresult], rax

    ; Terminate program via linux exit
    mov rax, __NR_exit      ; system call number (__NR_exit: 60)
    mov rdi, [outresult]    ; program result, error code (0 etc)
    syscall                 ; call kernel


