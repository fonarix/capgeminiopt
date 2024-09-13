; string reverse
;
; ./x64compile.sh task2x64.asm
; ./x64link.sh task2x64.o
; ./build/task2x64.out
; echo $?


%include "sys_codes.inc"
%include "unistd_64h.inc"


global _start                   ;must be declared for linker (gcc)

section .data
    str:        db  "This is a string!", 0  ; string
    str_len:    equ $-str                   ; Length of string

; Code goes in text section
section .text

_start:
    mov     rcx, str_len
    mov     rsi, str

    stackpush:
    mov     bx, [esi]
    push    bx
    inc     rsi
    loop    stackpush

    mov     rcx, str_len
    mov     rsi, str
    stackpop:
    pop     bx
    mov     [rsi], bl
    inc     rsi
    loop    stackpop

    ; print result
    mov     rsi, str            ;
    mov     rdx, str_len        ;
    mov     rax, __NR_write     ; system call command (__NR_write: 1)
    mov     rdi, fd_stdout      ; file descriptor (stdout: 1)
    syscall                     ; call kernel

    ; Terminate program via linux exit
    mov     rax, __NR_exit      ; system call number (__NR_exit: 60)
    mov     rdi, 0              ; program result, error code (0 etc)
    syscall                     ; call kernel



