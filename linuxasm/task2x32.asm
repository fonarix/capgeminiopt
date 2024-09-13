; string reverse
;
; ./x32compile.sh task1x32.asm
; ./x32link.sh task1x32.o
; ./build/task1x32.out
; echo $?


%include "sys_codes.inc"


global _start                   ;must be declared for linker (gcc)

section .data
    str:        db  "This is a string!", 0  ; string
    str_len:    equ $-str                   ; Length of string

section .bss
    str_result  resb 200

; Code goes in text section
section .text

_start:
    mov     ecx, str_len
    mov     esi, str

    call    upper

    stackpush:
    mov     bx, [esi]
    push    bx
    inc     esi
    loop    stackpush

    mov     ecx, str_len
    mov     esi, str_result
    stackpop:
    pop     bx
    mov     [esi], bl
    inc     esi
    loop    stackpop

    ; print result
    mov     ecx, str_result     ;
    mov     edx, str_len        ;
    mov     ebx, fd_stdout      ; file descriptor (stdout: 1)
    mov     eax, sys_write      ; system call number (sys_write: 4)
    int     80h                 ; call kernel

    ; Terminate program
    mov     eax, sys_exit       ; system call number (sys_exit: 1)
    mov     ebx, 0              ; program result
    int     80h                 ; call kernel

upper:
    push    ecx
    push    esi

    nextchar:

    cmp byte [esi+ecx], 'a'     ; compare if char code less 'a'
    jl      nextcharcontinue    ; continue
    cmp byte [esi+ecx], 'z'     ; compare if char code greater 'a'
    ja      nextcharcontinue    ; continue
    sub byte [esi+ecx], 0x20    ; Subtract 0x20 to get the uppercase Ascii value

    nextcharcontinue:           ; continue label
    loop    nextchar

    pop     esi
    pop     ecx
    ret



