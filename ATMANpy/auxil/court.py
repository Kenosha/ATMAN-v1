
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import Line, CourtLines
from ATMANpy.types.Shapes3D import Point3D
from ATMANpy.types.Visual3D import Camera, Scene, View

### COURT ###
courtLines = CourtLines.make().add_singles_netline()
courtScn = Scene.make_court()

### POINTS OF INTEREST ###
cnt_2D = courtLines.get_center_point()

top_deu_2D = courtLines.get_service_box(side = 'Top', which = 'Deuce').centroid()
top_ad_2D = courtLines.get_service_box(side = 'Top', which = 'Ad').centroid()
top_srv_2D = courtLines.get_service_box(side = 'Top', which = 'Deuce')['Top'].rightPt ### center of court, at service lines

bot_deu_2D = courtLines.get_service_box(side = 'Bottom', which = 'Deuce').centroid()
bot_ad_2D = courtLines.get_service_box(side = 'Bottom', which = 'Ad').centroid()
bot_srv_2D = courtLines.get_service_box(side = 'Bottom', which = 'Deuce')['Bottom'].leftPt ### center of court, at service lines

top_bsl_deu_2D = courtLines['Top Baseline'].leftPt
top_bsl_ad_2D = courtLines['Top Baseline'].rightPt
top_bsl_cnt_2D = courtLines['Top Baseline'].center()

bot_bsl_deu_2D = courtLines['Bottom Baseline'].rightPt
bot_bsl_ad_2D = courtLines['Bottom Baseline'].leftPt
bot_bsl_cnt_2D = courtLines['Bottom Baseline'].center()

top_srv_lft_2D = courtLines['Top Service Line'].leftPt
top_srv_rgt_2D = courtLines['Top Service Line'].rightPt
top_srv_cnt_2D = courtLines['Top Service Line'].center()

bot_srv_lft_2D = courtLines['Bottom Service Line'].leftPt
bot_srv_rgt_2D = courtLines['Bottom Service Line'].rightPt
bot_srv_cnt_2D = courtLines['Bottom Service Line'].center()

top_srv_ad_cnt_2D = Line(top_srv_cnt_2D, top_srv_rgt_2D).center()
top_srv_deu_cnt_2D = Line(top_srv_cnt_2D, top_srv_lft_2D).center()

bot_srv_deu_cnt_2D = Line(bot_srv_cnt_2D, bot_srv_rgt_2D).center()
bot_srv_ad_cnt_2D = Line(bot_srv_cnt_2D, bot_srv_lft_2D).center()

### CAMERAS ###
def birds_eye_service_box_cam(zoom = 1, which = 'Deuce', side = 'Top'):
    if side == 'Top' and which == 'Deuce':
        relevantPnt = top_deu_2D
    elif side == 'Top' and which == 'Ad':
        relevantPnt = top_ad_2D
    elif side == 'Bot' and which == 'Deuce':
        relevantPnt = bot_deu_2D
    elif side == 'Bot' and which == 'Ad':
        relevantPnt = bot_ad_2D
    
    c_w = Point3D.from_point2D(relevantPnt, z = 190).left(1) ### why we need to go one left is a mystery...
    f_w = Point3D.from_point2D(relevantPnt, z = 0)
    myCam = Camera.from_to(c_w, f_w, f = 1000 * zoom).rotate(-0.5)
    return myCam

def back_view_asymmetric_cam(which = 'Deuce', side = 'Top'):
    d_w = Point3D.from_point2D(cnt_2D, z = 0)
    if side == 'Top' and which == 'Deuce':
        c_w = Point3D.from_point2D(top_bsl_cnt_2D, z = 100).front(350).left(50)
    elif side == 'Top' and which == 'Ad':
        c_w = Point3D.from_point2D(top_bsl_cnt_2D, z = 100).front(350).right(50)
    elif side == 'Bot' and which == 'Deuce':
        c_w = Point3D.from_point2D(bot_bsl_cnt_2D, z = 100).back(350).right(50)
    elif side == 'Bot' and which == 'Ad':
        c_w = Point3D.from_point2D(bot_bsl_cnt_2D, z = 100).back(350).left(50)

    myCam = Camera.from_to(c_w, d_w, f = 1700)
    return myCam

def onto_service_box_cam(which = 'Deuce', side = 'Top', z = 100, dist = 350, f = 1700):
    if side == 'Top' and which == 'Deuce':
        c_w = Point3D.from_point2D(top_srv_deu_cnt_2D, z = z).front(dist)
        d_w = c_w.back(dist) ### dist doesn't matter here.
    elif side == 'Top' and which == 'Ad':
        c_w = Point3D.from_point2D(top_srv_ad_cnt_2D, z = 100).front(dist)
        d_w = c_w.back(dist) ### dist doesn't matter here.
    elif side == 'Bot' and which == 'Deuce':
        c_w = Point3D.from_point2D(bot_srv_deu_cnt_2D, z = 100).back(dist)
        d_w = c_w.front(dist) ### dist doesn't matter here.
    elif side == 'Bot' and which == 'Ad':
        c_w = Point3D.from_point2D(bot_srv_ad_cnt_2D, z = 100).back(dist)
        d_w = c_w.front(dist) ### dist doesn't matter here.

    myCam = Camera.from_to(c_w, d_w, f = f)
    return myCam

if __name__ == '__main__':
    top_deuCam = birds_eye_service_box_cam(zoom = 1, which = 'Deuce', side = 'Top')
    bot_deuCam = birds_eye_service_box_cam(zoom = 1, which = 'Deuce', side = 'Bot')
    top_adCam = birds_eye_service_box_cam(zoom = 1, which = 'Ad', side = 'Top')
    bot_adCam = birds_eye_service_box_cam(zoom = 1, which = 'Ad', side = 'Bot')

    top_back_deuCam = back_view_asymmetric_cam(which = 'Deuce', side = 'Top')
    top_back_adCam = back_view_asymmetric_cam(which = 'Ad', side = 'Top')

    View.make(top_deuCam, courtScn, resolution = 1).snap().show()
    View.make(top_adCam, courtScn, resolution = 1).snap().show()
    View.make(bot_deuCam, courtScn, resolution = 1).snap().show()
    View.make(bot_adCam, courtScn, resolution = 1).snap().show()
    View.make(top_back_deuCam, courtScn, resolution = 1).snap().show()
    View.make(top_back_adCam, courtScn, resolution = 1).snap().show()

    top_deuCam = onto_service_box_cam(which = 'Deuce', side = 'Top')
    top_adCam = onto_service_box_cam(which = 'Ad', side = 'Top')
    bot_deuCam = onto_service_box_cam(which = 'Deuce', side = 'Bot')
    bot_adCam = onto_service_box_cam(which = 'Ad', side = 'Bot')
    
    View.make(top_deuCam, courtScn, resolution = 1).snap().show()
    View.make(top_adCam, courtScn, resolution = 1).snap().show()
    View.make(bot_deuCam, courtScn, resolution = 1).snap().show()
    View.make(bot_adCam, courtScn, resolution = 1).snap().show()


