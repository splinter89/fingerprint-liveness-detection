<?php

$dd = [
    '../data-livdet-2015/Testing/Digital_Persona/Fake/',
    '../data-livdet-2015/Training/Digital_Persona/Fake/',
];
foreach ($dd as $d) {
    foreach (glob($d.'*/*.png') as $f) {
        $new_f = $d.str_replace([$d, '/', ' '], ['', '_', '_'], $f);
        echo $f."\n".$new_f."\n\n";
        //rename($f, $new_f);
    }
}
