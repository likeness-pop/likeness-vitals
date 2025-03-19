import likeness_vitals

###########################################################################

#                       IMPORTANT

#   Remove this file following the full removal of the ``v1.3.0`` namespace

###########################################################################


def test_constants_namespace():
    assert likeness_vitals.constants.GID == likeness_vitals.GID
    assert likeness_vitals.constants.BGID == likeness_vitals.BGID
    assert likeness_vitals.constants.BKID == likeness_vitals.BKID
    assert likeness_vitals.constants.PID == likeness_vitals.PID
    assert likeness_vitals.constants.HID == likeness_vitals.HID
    assert likeness_vitals.constants.XID == likeness_vitals.XID
    assert likeness_vitals.constants.CNT == likeness_vitals.CNT

    assert likeness_vitals.constants.SCL == likeness_vitals.SCL
    assert likeness_vitals.constants.TRS == likeness_vitals.TRS
    assert likeness_vitals.constants.EPSG_4326 == likeness_vitals.EPSG_4326
    assert likeness_vitals.constants.EPSG_3857 == likeness_vitals.EPSG_3857
