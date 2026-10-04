# scripts/v2/00_v1_freeze.ps1
# =============================================================================
# v1 을 v1.0-final 태그로 고정하고 v2 브랜치를 만든다.
#
# 순서: (1) main 이 원격과 같고 추적 파일 변경이 없는지 확인
#       (2) v1.0-final 태그 (이미 있으면 건너뜀) + push
#       (3) v2 브랜치 생성/전환 + push
# 추적되지 않은(untracked) 새 파일은 검사하지 않는다. v2 파일을 먼저 풀어 둬도 된다.
#
# 실행 (프로젝트 루트에서):
#   powershell -ExecutionPolicy Bypass -File scripts\v2\00_v1_freeze.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host "[중단] $m" -ForegroundColor Red; exit 1 }

Step "1. 저장소 상태 확인"
git rev-parse --is-inside-work-tree *> $null
if ($LASTEXITCODE -ne 0) { Fail "git 저장소 루트에서 실행하세요." }
$branch = (git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -ne "main" -and $branch -ne "v2") { Fail "현재 브랜치가 $branch 입니다. main 에서 실행하세요." }
$dirty = git status --porcelain --untracked-files=no
if ($dirty) { Fail "커밋 안 된 추적 파일 변경이 있습니다:`n$dirty" }
git fetch origin --tags
$local = (git rev-parse main).Trim()
$remote = (git rev-parse origin/main).Trim()
if ($local -ne $remote) { Fail "로컬 main($local) 과 origin/main($remote) 이 다릅니다. pull/push 후 다시 실행하세요." }
Write-Host "main = $local (원격과 동일)"

Step "2. v1.0-final 태그"
$tagExists = git tag --list "v1.0-final"
if ($tagExists) {
    Write-Host "이미 존재: v1.0-final -> $((git rev-list -n 1 v1.0-final).Trim())"
} else {
    git tag -a v1.0-final main -m "여유로 서울 v1 최종본 (1~8호선 쾌적 경로 추천, 배포: yeoyuro-seoul.streamlit.app)"
    git push origin v1.0-final
    Write-Host "생성·push 완료"
}

Step "3. v2 브랜치"
$hasV2 = git branch --list v2
if ($hasV2) { git switch v2 } else { git switch -c v2 main; git push -u origin v2 }
Write-Host "현재 브랜치: $((git rev-parse --abbrev-ref HEAD).Trim())"

Step "완료"
Write-Host "v1 은 v1.0-final 태그로 고정되었습니다. 이후 v2 작업은 v2 브랜치에서 합니다."
Write-Host "배포 앱(main) 은 v2 가 머지되기 전까지 v1 그대로 유지됩니다."
