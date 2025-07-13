#!/bin/zsh

# Claude Code Proxy Installation Script
# This script installs the Claude proxy globally for easy access

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_NAME="claude-code-proxy"
SCRIPT_NAME="claude-proxy"

# Get the current project directory
PROJECT_DIR="$(cd "$(dirname "${0}")" && pwd)"
BIN_DIR="$PROJECT_DIR/bin"
SCRIPT_PATH="$BIN_DIR/$SCRIPT_NAME"

# Detect the shell and get appropriate rc file
get_shell_rc() {
    local shell_name=$(basename "$SHELL")
    
    case "$shell_name" in
        zsh)
            echo "$HOME/.zshrc"
            ;;
        bash)
            if [[ -f "$HOME/.bash_profile" ]]; then
                echo "$HOME/.bash_profile"
            else
                echo "$HOME/.bashrc"
            fi
            ;;
        *)
            echo "$HOME/.profile"
            ;;
    esac
}

# Check if the script can be run
precheck() {
    echo -e "${BLUE}🔍 正在检查系统兼容性...${NC}"
    
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        echo -e "${RED}错误: 找不到主脚本文件 $SCRIPT_PATH${NC}"
        exit 1
    fi
    
    if [[ ! -x "$SCRIPT_PATH" ]]; then
        chmod +x "$SCRIPT_PATH"
        echo -e "${GREEN}已设置脚本执行权限${NC}"
    fi
    
    echo -e "${GREEN}✅ 预检查通过${NC}"
}

# Find a directory in PATH where we can write
create_symlink_path() {
    local suggested_dirs=("$HOME/.local/bin" "$HOME/bin" "/usr/local/bin")
    local target_dir=""
    
    # Check if we have permissions in suggested directories
    for dir in "$@"; do
        if [[ -d "$dir" ]] && [[ -w "$dir" ]]; then
            target_dir="$dir"
            break
        fi
    done
    
    # If no suitable directory found, create ~/.local/bin
    if [[ -z "$target_dir" ]]; then
        target_dir="$HOME/.local/bin"
        if [[ ! -d "$target_dir" ]]; then
            mkdir -p "$target_dir"
            echo -e "${YELLOW}创建目录: $target_dir${NC}"
        fi
    fi
    
    echo "$target_dir"
}

# Add directory to PATH if not already in PATH
add_to_path() {
    local target_dir=$1
    local rc_file=$2
    
    if [[ ":$PATH:" != *":$target_dir:"* ]]; then
        echo "" >> "$rc_file"
        echo "# Claude Code Proxy - Add $target_dir to PATH" >> "$rc_file"
        echo "export PATH=\"$target_dir:\$PATH\"" >> "$rc_file"
        echo -e "${GREEN}已将 $target_dir 添加到 PATH (写入 $rc_file)${NC}"
        return 0
    fi
    
    return 1
}

# Install the script
install_script() {
    echo -e "${BLUE}📦 正在安装 Claude Code Proxy...${NC}"
    
    local target_dir=$(create_symlink_path)
    local symlink_path="$target_dir/$SCRIPT_NAME"
    
    # Check if symlink already exists
    if [[ -L "$symlink_path" ]] || [[ -f "$symlink_path" ]]; then
        echo -e "${YELLOW}⚠️  $symlink_path 已存在，将被替换${NC}"
        rm -f "$symlink_path"
    fi
    
    # Create symlink
    ln -sf "$SCRIPT_PATH" "$symlink_path"
    echo -e "${GREEN}✅ 已创建符号链接: $symlink_path -> $SCRIPT_PATH${NC}"
    
    # Add to PATH if necessary
    local rc_file=$(get_shell_rc)
    if add_to_path "$target_dir" "$rc_file"; then
        echo -e "${YELLOW}🔔 请将 $target_dir 添加到 PATH，或重新加载 shell${NC}"
        echo -e "${YELLOW}   你可以运行: source $rc_file${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}🎉 安装完成!${NC}"
    echo ""
    echo -e "${BLUE}使用说明:${NC}"
    echo "  • 启动服务器: $SCRIPT_NAME"
    echo "  • 开发模式: $SCRIPT_NAME -d"
    echo "  • 指定端口: $SCRIPT_NAME -p 8080"
    echo "  • Docker启动: $SCRIPT_NAME --docker"
    echo "  • 查看帮助: $SCRIPT_NAME --help"
    echo ""
    echo -e "${YELLOW}配置提示:${NC}"
    echo "  1. 在项目目录编辑 .env 文件配置 API 密钥"
    echo "  2. 修改 models.yaml 配置模型设置"
    echo "  3. 项目目录: $PROJECT_DIR"
}

# Main installation process
main() {
    echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          Claude Code Proxy 安装器                ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
    
    precheck
    install_script
    
    echo -e "${GREEN}现在你可以在任何目录使用 $SCRIPT_NAME 命令!${NC}"
}

# Handle special cases
if [[ "$1" == "--check" ]]; then
    if [[ -L "/usr/local/bin/$SCRIPT_NAME" ]] || [[ -f "$HOME/.local/bin/$SCRIPT_NAME" ]] || [[ -f "$HOME/bin/$SCRIPT_NAME" ]]; then
        echo -e "${GREEN}已安装${NC}"
        exit 0
    else
        echo -e "${RED}未安装${NC}"
        exit 1
    fi
fi

# Run main installation
main "$@"