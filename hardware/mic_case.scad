// Airacle — 차량 지붕 마이크 케이스 (빵모자/버섯형)  v2 파라메트릭
// ReSpeaker 마이크 어레이용 루프탑 인클로저. 2파트(베이스 트레이 + 빵모자 캡, 텔레스코핑).
// OpenSCAD: F5 미리보기 → F6 렌더 → 각 파트 STL 내보내기.
//
// ⚠ v2는 로컬 미검증 초안 — F5로 단면 확인 후 아래 값 미세조정하세요.
//   - board_d : 보드 실측 지름(현재 75)
//   - base_h  : 보드+USB 커넥터가 들어갈 내부 높이(커넥터 높이보다 크게)
//   - fit_gap : 캡이 베이스에 헐겁/빡빡하면 조정(프린터 공차)
//   - crown_d : 크라운 최대 지름(밴드보다 커야 '처마' 생김)
// 형태 근거(대화): 보드 Ø75, 높이 45, 크라운 Ø100(밴드보다 밖으로=처마), 자석, 측면 케이블.
// 참고: 로브(봉우리)는 물고임·서포트·폼 문제로 생략(매끈 버섯). 처마가 포트를 빗물서 가림.
//   포트는 하향, 안쪽에 폼 윈드스크린 필수(풍절음). 회전대칭이라 DOA(4마이크 방향) 유지.

$fn = 140;

/* ================= 파라미터 (mm) ================= */
board_d   = 75;      // 보드 지름(측정)
clr       = 1.0;     // 보드 편측 여유
wall      = 2.6;     // 벽 두께
case_h    = 45;      // 캡(빵모자) 높이
floor_t   = 3.0;     // 베이스 바닥 두께
base_h    = 13;      // 베이스 트레이 높이(보드+커넥터 여유) ⚠ 커넥터 높이 확인
band_h    = 11;      // 캡 밴드(스커트) 높이 — 베이스 위로 슬립
fit_gap   = 0.4;     // 캡-베이스 끼움 공차
crown_d   = 100;     // 크라운 최대 지름(밴드보다 큼 = 처마)

button    = true;    // 꼭지
button_r  = 4;
cable_d   = 6.5;     // 측면 케이블 관통 지름
port_n    = 20;      // 음향 포트 수
port_d    = 3.2;     // 포트 지름
port_z    = 5;       // 포트 높이(밴드 상)
port_tilt = 15;      // 포트 하향각(빗물 배제)
magnet_d  = 10;      // 자석 지름
magnet_h  = 2.6;     // 자석 포켓 깊이
magnet_n  = 3;       // 자석 개수

/* ================= 파생 ================= */
inner_r = board_d/2 + clr;     // 보드 수용 안반경 ≈ 38.5
base_r  = inner_r + wall;      // 베이스 외반경  ≈ 41.1 (외경 ~82)
cap_ir  = base_r + fit_gap;    // 캡 밴드 안반경 ≈ 41.5
band_or = cap_ir + wall;       // 밴드 외반경   ≈ 44.1 (외경 ~88)
crown_r = crown_d/2;           // 크라운 외반경 = 50
mag_pos = inner_r * 0.6;

/* ================= 베이스 트레이 ================= */
// 보드 안착 + 측면 케이블 입구 + 바닥 자석 포켓. 캡이 이 위로 슬립.
module base_tray(){
  difference(){
    cylinder(h=base_h, r=base_r);
    translate([0,0,floor_t]) cylinder(h=base_h, r=inner_r);            // 보드 캐비티
    translate([0,0,floor_t+3]) rotate([0,90,0])
      cylinder(h=base_r+2, d=cable_d);                                 // 측면 케이블
    for(i=[0:magnet_n-1]) rotate([0,0,360/magnet_n*i])
      translate([mag_pos,0,-0.01]) cylinder(h=magnet_h, d=magnet_d);   // 바닥 자석 포켓
  }
}

/* ================= 캡(빵모자/버섯) ================= */
// 회전체 단면 [반경,높이]. 밴드(수직) → 밖으로 부풀어 처마 → 둥근 크라운 → 꼭지.
outer = [
  [0,0],[band_or,0],[band_or,band_h],
  [crown_r,     band_h+7],       // ← 밖으로 부풀기(처마)
  [crown_r-2,   band_h+15],
  [crown_r-11,  case_h-9],
  [crown_r-24,  case_h-2],
  [0,           case_h]
];
inner = [
  [0,0],[cap_ir,0],[cap_ir,band_h],
  [crown_r-wall,      band_h+7],
  [crown_r-2-wall,    band_h+15],
  [crown_r-11-wall,   case_h-9],
  [crown_r-24-wall,   case_h-2-wall],
  [0,                 case_h-wall]
];
module cap(){
  difference(){
    union(){
      rotate_extrude() polygon(outer);
      if(button) translate([0,0,case_h-0.5]) sphere(button_r);
    }
    rotate_extrude() polygon(inner);                                   // 셸 비우기(바닥 개방)
    for(i=[0:port_n-1]) rotate([0,0,360/port_n*i])                     // 음향 포트(하향, 처마 밑)
      translate([band_or+1,0,port_z]) rotate([0,90+port_tilt,0])
        cylinder(h=wall+4, d=port_d, center=true);
  }
}

/* ================= 배치 ================= */
base_tray();
translate([2*crown_r+18, 0, 0]) cap();          // 나란히(프린트용)

// 조립 미리보기: 위 두 줄 주석 후 아래 사용
// base_tray();
// translate([0,0, base_h-band_h]) cap();
