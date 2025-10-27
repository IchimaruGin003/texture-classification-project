#!/bin/sh
set -e

# 如果第一个参数以"--"开头，那么执行python app.py并传递所有参数
if [ "${1#-}" != "$1" ]; then
    set -- python src/app.py "$@"
fi

# 执行命令
exec "$@"