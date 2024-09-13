;
; BIOS interruptions defines
;
; Name              %eax    %ebx            %ecx            %edx

sys_exit    EQU     1       ;int
sys_fork    EQU     2       ;struct pt_regs
sys_read    EQU     3       ;unsigned int   char * 	        size_t
sys_write   EQU     4       ;unsigned int   const char *    size_t
sys_open    EQU     5       ;const char *   int             int
sys_close   EQU     6       ;unsigned int


; sys_write descriptors
fd_stdin    EQU     0       ;std in
fd_stdout   EQU     1       ;stdout
fd_stderr   EQU     2



