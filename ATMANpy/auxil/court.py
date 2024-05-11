
from ATMANpy.types.Points import Point2D
from ATMANpy.types.Lines import CourtLines
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

### CAMERAS ###
c_w = Point3D.from_point2D(top_deu_2D, z = 190).left(1) ### why we need to go one left is a mystery...
f_w = Point3D.from_point2D(top_deu_2D, z = 0)
top_deuCam = Camera.from_to(c_w, f_w, f = 1000).rotate(-0.5)

c_w = Point3D.from_point2D(top_ad_2D, z = 180).left(1) ### why we need to go one left is a mystery...
f_w = Point3D.from_point2D(top_ad_2D, z = 0)
top_adCam = Camera.from_to(c_w, f_w, f = 1000).rotate(-0.5)

c_w = Point3D.from_point2D(bot_deu_2D, z = 190).left(1) ### why we need to go one left is a mystery...
f_w = Point3D.from_point2D(bot_deu_2D, z = 0)
bot_deuCam = Camera.from_to(c_w, f_w, f = 1000).rotate(-0.5)

c_w = Point3D.from_point2D(bot_ad_2D, z = 180).left(1) ### why we need to go one left is a mystery...
f_w = Point3D.from_point2D(bot_ad_2D, z = 0)
bot_adCam = Camera.from_to(c_w, f_w, f = 1000).rotate(-0.5)

c_w = Point3D.from_point2D(top_bsl_cnt_2D, z = 100).front(350).left(50)
d_w = Point3D.from_point2D(cnt_2D, z = 0)
top_back_deuCam = Camera.from_to(c_w, d_w, f = 1700)

c_w = Point3D.from_point2D(top_bsl_cnt_2D, z = 100).front(350).right(50)
d_w = Point3D.from_point2D(cnt_2D, z = 0)
top_back_adCam = Camera.from_to(c_w, d_w, f = 1700)


if __name__ == '__main__':
    View.make(top_deuCam, courtScn, resolution = 1).snap().show()
    View.make(top_adCam, courtScn, resolution = 1).snap().show()
    View.make(bot_deuCam, courtScn, resolution = 1).snap().show()
    View.make(bot_adCam, courtScn, resolution = 1).snap().show()
    View.make(top_back_deuCam, courtScn, resolution = 1).snap().show()
    View.make(top_back_adCam, courtScn, resolution = 1).snap().show()
