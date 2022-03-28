set nocompatible
filetype on
filetype indent on
filetype plugin on
filetype plugin indent on
set mouse=a
set encoding=utf-8
let &t_ut=''
let mapleader = " "
set tabstop=4
set number
set relativenumber
set cursorline
set wildmenu
set hlsearch
exec "nohlsearch"
set incsearch
set ignorecase
set smartcase
set foldmethod=indent
set foldlevel=99
set updatetime=100
set nowrap
set scrolloff=5

let &t_SI = "\<Esc>]50;CursorShape=1\x7"
let &t_SR = "\<Esc>]50;CursorShape=2\x7"
let &t_EI = "\<Esc>]50;CursorShape=0\x7"
au BufReadPost * if line("'\"") > 1 && line("'\"") <= line("$") | exe "normal! g'\"" | endif

syntax on

noremap <LEADER><CR> :nohlsearch<CR>
noremap J            5j
noremap K            5k
noremap W            5w
noremap B            5b

map s <nop>
map S :w<CR>
map Q :q<CR>
map R :source ~/.config/nvim/init.vim<CR>
map <LEADER>vi :e ~/.config/nvim/init.vim<CR>
map <LEADER>zi :e ~/.zshrc<CR>

"splite and move among different windows
map sl  :set splitright<CR>:vsplit<CR>
map sh  :set nosplitright<CR>:vsplit<CR>
map sk  :set nosplitbelow<CR>:split<CR>
map sj  :set splitbelow<CR>:split<CR>
map stl :set splitright<CR>:vsplit<CR>:terminal<CR>

map <LEADER>l <C-w>l
map <LEADER>k <C-w>k
map <LEADER>h <C-w>h
map <LEADER>j <C-w>j
map <LEADER>r :RnvimrToggle<CR>

map <up>    :res      +5<CR>
map <down>  :res      -5<CR>
map <left>  :vertical resize+5<CR>
map <right> :vertical resize-5<CR>

" tas
map th :-tabnext<CR>
map tl :+tabnext<CR>
map tn :tabnew<CR>
map tm :terminal<CR>

" NERDTree
map tt :NERDTree<CR>

map sv <C-w>t<C-w>H
map sh <C-w>t<C-w>K

tnoremap <Esc> <C-\><C-n>
call plug#begin('~/.config/nvim/plugged')

Plug 'vim-airline/vim-airline'
Plug 'preservim/nerdtree'
Plug 'mbbill/undotree'
Plug 'preservim/nerdcommenter'

" ranger support within nvim
Plug 'kevinhwang91/rnvimr'

"Plug 'neoclide/coc.nvim', {'branch': 'release'}

" fzf support
Plug 'junegunn/fzf' 
Plug 'junegunn/fzf.vim'
" find and replace 
Plug 'rking/ag.vim'
Plug 'brooth/far.vim'

" write code support
Plug 'tpope/vim-surround'
Plug 'junegunn/vim-easy-align'
Plug 'kshenoy/vim-signature'
Plug 'tmhedberg/SimpylFold'
Plug 'mg979/vim-visual-multi', {'branch': 'master'}

Plug 'mg979/vim-xtabline'

" easycomplete is a simple complete Plug for vim
Plug 'jayli/vim-easycomplete'

" git spport
Plug 'mhinz/vim-signify'
Plug 'tpope/vim-fugitive'
Plug 'rbong/vim-flog'

"Plug 'liuchengxu/vista.vim'

" cython syntax support
Plug 'tshirtman/vim-cython'

" tagbar
"Plug 'preservim/tagbar'

" snazzy theme
Plug 'connorholyday/vim-snazzy'

Plug 'numirias/semshi'
call plug#end()

" select color theme: snazzy
color snazzy

" remap ra to open ranger
noremap ra :RnvimrToggle<CR>

" easycomplete setting
noremap gr :EasyCompleteReference<CR>
noremap gd :EasyCompleteGotoDefinition<CR>
noremap rn :EasyCompleteRename<CR>

" vim-easy-align settings
xmap ga <Plug>(EasyAlign)
nmap ga <Plug>(EasyAlign)

" ag.vim setting
let g:ag_prg="ag --vimgrep --smart-case"
" search slelcted word under current folder
noremap <LEADER>ag :Ag! -w "<cword>"<CR>
" search slelcted word under current file
noremap <LEADER>af :Ag! -w "<cword>" %<CR>

" fzf file finder setting
noremap <C-p> :Files<CR>
noremap <C-h> :History<CR>
noremap <C-b> :History<CR>

" git related plugin setting
"" signify setting
let g:signify_sign_add               = '+'
let g:signify_sign_delete            = '_'
let g:signify_sign_delete_first_line = '‾'
let g:signify_sign_change            = '~'
highlight SignifySignAdd    ctermfg=green  guifg=#00ff00 cterm=NONE gui=NONE
highlight SignifySignDelete ctermfg=red    guifg=#ff0000 cterm=NONE gui=NONE
highlight SignifySignChange ctermfg=yellow guifg=#ffff00 cterm=NONE gui=NONE

nmap <leader>gj <plug>(signify-next-hunk)
nmap <leader>gk <plug>(signify-prev-hunk)
nmap <leader>gJ 9999<leader>gj
nmap <leader>gK 9999<leader>gk

noremap <LEADER>gd :SignifyHunkDiff<CR>
noremap <LEADER>gds :Gdiffsplit<CR>
"" GitGutter setting
noremap <LEADER>gf :GitGutterFold<CR>
