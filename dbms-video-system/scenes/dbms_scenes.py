"""
Manim animations for the "DBMS Made Simple" series.

Each class below is one short animated explainer (15-40 s).
Episodes in config/episodes.yaml pick them by class name.

Render one scene by hand (low quality preview):
    manim -pql scenes/dbms_scenes.py Normalization
Or let the pipeline do it:
    python vedit.py scenes U2-05

Only plain Text is used (no LaTeX), so no TeX installation is needed.
"""
from pathlib import Path

import yaml
from manim import *  # noqa: F401,F403

# ------------------------------------------------------------------ theme
_THEME = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "theme.yaml").read_text(encoding="utf-8"))
C = _THEME["colors"]
BG, SURFACE, TXT, MUTED = C["background"], C["surface"], C["text"], C["muted"]
PRIMARY, ACCENT, GOOD, BAD = C["primary"], C["accent"], C["success"], C["danger"]
HEAD_FONT = _THEME["fonts"]["heading"]
BODY_FONT = _THEME["fonts"]["body"]
MONO_FONT = _THEME["fonts"]["mono"]
UNIT_COLOR = {str(k): v for k, v in _THEME["unit_colors"].items()}

config.background_color = BG


# ------------------------------------------------------------------ helpers
def T(text, size=28, color=TXT, font=BODY_FONT, bold=False):
    return Text(text, font=font, font_size=size, color=color, weight=BOLD if bold else NORMAL)


def mono(text, size=24, color=TXT):
    return Text(text, font=MONO_FONT, font_size=size, color=color)


def fit(mob, max_w):
    if mob.width > max_w:
        mob.scale_to_fit_width(max_w)
    return mob


def box(label, w=3.0, h=1.0, color=PRIMARY, size=26, fill=SURFACE, text_color=TXT):
    rect = RoundedRectangle(corner_radius=0.15, width=w, height=h, stroke_color=color,
                            stroke_width=3, fill_color=fill, fill_opacity=1)
    t = fit(T(label, size, text_color), w - 0.3)
    t.move_to(rect)
    return VGroup(rect, t)


def cylinder(label="DB", w=1.8, h=1.6, color=PRIMARY):
    body = Rectangle(width=w, height=h, stroke_width=0, fill_color=SURFACE, fill_opacity=1)
    top = Ellipse(width=w, height=0.45, stroke_color=color, stroke_width=3, fill_color=SURFACE, fill_opacity=1)
    bottom = Ellipse(width=w, height=0.45, stroke_color=color, stroke_width=3, fill_color=SURFACE, fill_opacity=1)
    top.move_to(body.get_top())
    bottom.move_to(body.get_bottom())
    sides = VGroup(Line(body.get_corner(UL), body.get_corner(DL)), Line(body.get_corner(UR), body.get_corner(DR)))
    sides.set_stroke(color, 3)
    t = fit(T(label, 24), w - 0.2).move_to(body)
    return VGroup(bottom, body, sides, top, t)


def grid(rows, col_w=1.8, row_h=0.55, head_color=PRIMARY, size=22, key_cols=()):
    """A simple table. Returns VGroup of rows; table[r][c] is one cell (VGroup(rect, text))."""
    widths = col_w if isinstance(col_w, (list, tuple)) else [col_w] * len(rows[0])
    table = VGroup()
    for r, row in enumerate(rows):
        row_g = VGroup()
        x = 0.0
        for c, val in enumerate(row):
            w = widths[c]
            head = r == 0
            rect = Rectangle(width=w, height=row_h, stroke_color=MUTED, stroke_width=1.5,
                             fill_color=head_color if head else SURFACE, fill_opacity=0.25 if head else 1)
            rect.move_to([x + w / 2, -r * row_h, 0])
            colour = head_color if head else TXT
            if head and c in key_cols:
                colour = ACCENT
            t = fit(T(str(val), size, colour, bold=head), w - 0.15).move_to(rect)
            cell = VGroup(rect, t)
            if head and c in key_cols:
                cell.add(Underline(t, color=ACCENT, buff=0.05))
            row_g.add(cell)
            x += w
        table.add(row_g)
    return table.center()


def column(table, c):
    return VGroup(*[row[c] for row in table])


def ellipse_attr(label, key=False, multi=False, derived=False, w=1.9, h=0.75):
    e = Ellipse(width=w, height=h, stroke_color=PRIMARY, stroke_width=2.5, fill_color=SURFACE, fill_opacity=1)
    parts = [e]
    if multi:
        parts.append(Ellipse(width=w - 0.18, height=h - 0.15, stroke_color=PRIMARY, stroke_width=2))
    if derived:
        parts = [DashedVMobject(e, num_dashes=24)]
    t = fit(T(label, 20), w - 0.3).move_to(e)
    parts.append(t)
    if key:
        parts.append(Underline(t, color=ACCENT, buff=0.04))
    return VGroup(*parts)


def diamond(label, w=2.4, h=1.3, color=ACCENT):
    d = Polygon([0, h / 2, 0], [w / 2, 0, 0], [0, -h / 2, 0], [-w / 2, 0, 0],
                stroke_color=color, stroke_width=3, fill_color=SURFACE, fill_opacity=1)
    return VGroup(d, fit(T(label, 20), w * 0.6).move_to(d))


def entity(label, w=2.4, h=0.9, color=PRIMARY):
    return box(label, w, h, color, 26)


def connect(a, b, color=MUTED, width=2.5):
    return Line(a.get_center(), b.get_center(), color=color, stroke_width=width).set_z_index(-1)


class ThemedScene(Scene):
    unit = "1"

    def header(self, title, sub=None):
        colour = UNIT_COLOR.get(str(self.unit), PRIMARY)
        t = T(title, 40, TXT, HEAD_FONT, bold=True).to_corner(UL, buff=0.5)
        bar = Rectangle(width=0.12, height=t.height + 0.1, fill_color=colour, fill_opacity=1, stroke_width=0)
        bar.next_to(t, LEFT, buff=0.2)
        g = VGroup(bar, t)
        if sub:
            s = T(sub, 24, MUTED).next_to(t, DOWN, aligned_edge=LEFT, buff=0.15)
            g.add(s)
        self.play(FadeIn(g, shift=RIGHT * 0.3), run_time=0.8)
        return g

    def caption(self, text, old=None, color=TXT):
        cap = fit(T(text, 28, color), 12.5).to_edge(DOWN, buff=0.45)
        if old is None:
            self.play(FadeIn(cap, shift=UP * 0.2), run_time=0.6)
        else:
            self.play(FadeOut(old, shift=UP * 0.2), FadeIn(cap, shift=UP * 0.2), run_time=0.6)
        return cap

    def outro(self):
        self.wait(1.5)
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


# ================================================================== UNIT I
class FileSystemVsDBMS(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("File System vs DBMS")
        apps = ["Accounts", "Library", "Exams"]
        left_title = T("File processing", 30, BAD, bold=True).move_to([-3.6, 2.0, 0])
        right_title = T("Database system", 30, GOOD, bold=True).move_to([3.6, 2.0, 0])
        divider = DashedLine([0, 2.4, 0], [0, -2.6, 0], color=MUTED)
        self.play(FadeIn(left_title), FadeIn(right_title), Create(divider))

        left = VGroup()
        for i, name in enumerate(apps):
            app = box(name, 2.0, 0.7, BAD, 22).move_to([-5.0, 1.0 - i * 1.25, 0])
            f = box(f"{name.lower()}.csv", 2.3, 0.6, MUTED, 18).move_to([-2.2, 1.0 - i * 1.25, 0])
            left.add(VGroup(app, f, Arrow(app.get_right(), f.get_left(), buff=0.1, color=MUTED)))
        self.play(LaggedStart(*[FadeIn(g) for g in left], lag_ratio=0.3))
        dup = T('"Asha, Pune" stored 3 times', 22, ACCENT).move_to([-3.6, -2.6, 0])
        self.play(Write(dup))
        self.play(*[Indicate(g[1], color=BAD) for g in left])

        db = cylinder("Student DB", 2.2, 1.8, GOOD).move_to([4.6, -0.3, 0])
        right = VGroup()
        for i, name in enumerate(apps):
            app = box(name, 2.0, 0.7, GOOD, 22).move_to([1.6, 1.0 - i * 1.25, 0])
            right.add(VGroup(app, Arrow(app.get_right(), db.get_left(), buff=0.15, color=GOOD)))
        self.play(FadeIn(db), LaggedStart(*[FadeIn(g) for g in right], lag_ratio=0.3))
        one = T("One shared, consistent copy", 22, GOOD).move_to([3.6, -2.6, 0])
        self.play(Write(one))

        problems = ["Redundancy", "Inconsistency", "No concurrency control", "Weak security"]
        cap = None
        for p in problems:
            cap = self.caption(f"File systems suffer from: {p}  ->  a DBMS solves it", cap)
            self.wait(1.2)
        self.outro()


class ThreeSchemaArchitecture(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("Three-Schema Architecture", "Data abstraction in three levels")
        views = VGroup(*[box(f"View {i}", 2.2, 0.75, PRIMARY, 22) for i in (1, 2, 3)]).arrange(RIGHT, buff=0.5)
        views.move_to([0.8, 1.6, 0])
        conceptual = box("Conceptual schema  (all tables, keys, constraints)", 8.0, 0.8, ACCENT, 22).move_to([0.8, 0.0, 0])
        internal = box("Internal schema  (files, indexes, storage)", 8.0, 0.8, GOOD, 22).move_to([0.8, -1.6, 0])
        disk = cylinder("Disk", 1.6, 0.9, MUTED).move_to([0.8, -3.0, 0]).scale(0.8)

        labels = VGroup(
            T("External", 24, PRIMARY, bold=True).move_to([-5.2, 1.6, 0]),
            T("Conceptual", 24, ACCENT, bold=True).move_to([-5.2, 0.0, 0]),
            T("Internal", 24, GOOD, bold=True).move_to([-5.2, -1.6, 0]),
        )
        self.play(FadeIn(labels[0]), LaggedStart(*[FadeIn(v, shift=DOWN * 0.2) for v in views], lag_ratio=0.2))
        cap = self.caption("External level: each user sees only their own view")
        self.wait(1)
        arrows1 = VGroup(*[Arrow(v.get_bottom(), conceptual.get_top(), buff=0.08, color=MUTED) for v in views])
        self.play(FadeIn(labels[1]), FadeIn(conceptual), *[GrowArrow(a) for a in arrows1])
        cap = self.caption("Conceptual level: the complete logical design of the database", cap)
        self.wait(1)
        arrow2 = Arrow(conceptual.get_bottom(), internal.get_top(), buff=0.08, color=MUTED)
        self.play(FadeIn(labels[2]), FadeIn(internal), GrowArrow(arrow2), FadeIn(disk))
        cap = self.caption("Internal level: how the data is physically stored", cap)
        self.wait(1)

        logical = T("Logical data independence", 20, PRIMARY).next_to(arrows1, RIGHT, buff=0.3).shift(RIGHT * 0.2)
        physical = T("Physical data independence", 20, GOOD).next_to(arrow2, RIGHT, buff=2.6)
        self.play(Write(logical))
        cap = self.caption("Change the conceptual schema without rewriting user views", cap)
        self.wait(1.2)
        self.play(Write(physical))
        cap = self.caption("Change storage (add an index, move files) without changing tables", cap)
        self.outro()


class DataModels(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("Data Models")
        positions = [[-3.4, 1.0, 0], [3.4, 1.0, 0], [-3.4, -2.0, 0], [3.4, -2.0, 0]]

        # Hierarchical: a tree
        root = Circle(0.25, color=PRIMARY, fill_opacity=1, fill_color=SURFACE)
        kids = VGroup(*[Circle(0.22, color=PRIMARY, fill_opacity=1, fill_color=SURFACE) for _ in range(3)]).arrange(RIGHT, buff=0.6)
        kids.next_to(root, DOWN, buff=0.7)
        tree = VGroup(*[connect(root, k) for k in kids], root, kids)
        hier = VGroup(tree, T("Hierarchical: tree, one parent", 22)).arrange(DOWN, buff=0.3)

        # Network: graph with shared children
        nodes = VGroup(*[Circle(0.22, color=ACCENT, fill_opacity=1, fill_color=SURFACE) for _ in range(5)])
        for n, p in zip(nodes, [[-1, 0.6, 0], [1, 0.6, 0], [-1.5, -0.6, 0], [0, -0.6, 0], [1.5, -0.6, 0]]):
            n.move_to(p)
        edges = VGroup(*[connect(nodes[a], nodes[b]) for a, b in [(0, 2), (0, 3), (1, 3), (1, 4), (0, 4)]])
        net = VGroup(VGroup(edges, nodes), T("Network: graph, many parents", 22)).arrange(DOWN, buff=0.3)

        # Relational: table
        rel = VGroup(grid([["ID", "Name", "Dept"], ["1", "Asha", "ECE"], ["2", "Ravi", "CS"]], 1.1, 0.42, GOOD, 18),
                     T("Relational: tables + keys", 22)).arrange(DOWN, buff=0.3)

        # Object-oriented: class box
        cls = VGroup(box("Student", 2.6, 0.5, BAD, 20), box("name, cgpa", 2.6, 0.5, MUTED, 18),
                     box("getGrade()", 2.6, 0.5, MUTED, 18)).arrange(DOWN, buff=0)
        oo = VGroup(cls, T("Object-oriented: data + methods", 22)).arrange(DOWN, buff=0.3)

        for g, p in zip([hier, net, rel, oo], positions):
            g.move_to(p)
            self.play(FadeIn(g, shift=UP * 0.2), run_time=0.9)
            self.wait(1.5)
        self.play(Circumscribe(rel, color=GOOD))
        self.caption("This course focuses on the relational model", color=GOOD)
        self.outro()


class DBMSComponents(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("Inside a DBMS")
        user = box("User / App", 2.2, 0.8, MUTED, 22).move_to([-5.4, 0.4, 0])
        qp = VGroup(T("Query Processor", 26, PRIMARY, bold=True),
                    *[T(s, 20) for s in ["DDL interpreter", "DML compiler", "Query optimiser", "Evaluation engine"]]
                    ).arrange(DOWN, buff=0.2)
        qp_box = SurroundingRectangle(qp, color=PRIMARY, buff=0.3, corner_radius=0.15)
        qp_g = VGroup(qp_box, qp).move_to([-1.6, 0.4, 0])
        sm = VGroup(T("Storage Manager", 26, ACCENT, bold=True),
                    *[T(s, 20) for s in ["Authorisation", "Transaction manager", "File manager", "Buffer manager"]]
                    ).arrange(DOWN, buff=0.2)
        sm_box = SurroundingRectangle(sm, color=ACCENT, buff=0.3, corner_radius=0.15)
        sm_g = VGroup(sm_box, sm).move_to([2.4, 0.4, 0])
        disk = cylinder("Data + Indexes\n+ Dictionary", 2.2, 1.8, GOOD).move_to([5.6, 0.4, 0])

        a1 = Arrow(user.get_right(), qp_g.get_left(), color=MUTED)
        a2 = Arrow(qp_g.get_right(), sm_g.get_left(), color=MUTED)
        a3 = Arrow(sm_g.get_right(), disk.get_left(), color=MUTED)
        self.play(FadeIn(user))
        self.play(GrowArrow(a1), FadeIn(qp_g))
        self.play(GrowArrow(a2), FadeIn(sm_g))
        self.play(GrowArrow(a3), FadeIn(disk))

        q = mono("SELECT * FROM student;", 20, ACCENT).next_to(user, UP, buff=0.3)
        self.play(Write(q))
        cap = self.caption("1. The query processor checks the SQL and picks the fastest plan")
        self.play(q.animate.next_to(qp_g, UP, buff=0.2), run_time=1.2)
        self.wait(0.8)
        cap = self.caption("2. The storage manager fetches pages through the buffer", cap)
        self.play(q.animate.next_to(sm_g, UP, buff=0.2), run_time=1.2)
        self.wait(0.8)
        cap = self.caption("3. Data comes off the disk and the result goes back to the user", cap)
        self.play(Indicate(disk, color=GOOD))
        self.outro()


class ERDiagramBasics(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("ER Diagram Basics", "University example")
        student = entity("STUDENT").move_to([-4.0, -0.3, 0])
        course = entity("COURSE").move_to([4.0, -0.3, 0])
        enrolls = diamond("ENROLLS").move_to([0, -0.3, 0])

        s_attrs = [ellipse_attr("Roll_No", key=True).move_to([-5.6, 1.5, 0]),
                   ellipse_attr("Name").move_to([-3.4, 1.7, 0]),
                   ellipse_attr("Phone", multi=True).move_to([-5.6, -2.2, 0]),
                   ellipse_attr("Age", derived=True).move_to([-3.2, -2.3, 0])]
        c_attrs = [ellipse_attr("Course_ID", key=True).move_to([3.0, 1.6, 0]),
                   ellipse_attr("Title").move_to([5.3, 1.6, 0]),
                   ellipse_attr("Credits").move_to([4.8, -2.2, 0])]
        grade = ellipse_attr("Grade").move_to([0, -2.2, 0])

        self.play(FadeIn(student), FadeIn(course))
        cap = self.caption("Rectangle = entity (a real-world thing)")
        self.wait(0.8)
        self.play(*[FadeIn(a) for a in s_attrs + c_attrs], *[Create(connect(student, a)) for a in s_attrs],
                  *[Create(connect(course, a)) for a in c_attrs])
        cap = self.caption("Ellipse = attribute.  Underlined = key,  double = multivalued,  dashed = derived", cap)
        self.wait(2)
        self.play(FadeIn(enrolls), Create(connect(student, enrolls)), Create(connect(enrolls, course)),
                  FadeIn(grade), Create(connect(enrolls, grade)))
        cap = self.caption("Diamond = relationship (it can have its own attributes, like Grade)", cap)
        self.wait(1.5)
        m = T("M", 30, ACCENT, bold=True).move_to([-2.2, 0.05, 0])
        n = T("N", 30, ACCENT, bold=True).move_to([2.2, 0.05, 0])
        self.play(Write(m), Write(n))
        cap = self.caption("Cardinality M:N: a student takes many courses, a course has many students", cap)
        self.outro()


class EERSpecialization(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("EER: Specialization & Generalization")
        person = entity("PERSON").move_to([0, 1.6, 0])
        pid = ellipse_attr("ID", key=True).move_to([-2.6, 2.1, 0])
        pname = ellipse_attr("Name").move_to([2.6, 2.1, 0])
        d = Circle(0.32, color=ACCENT, fill_color=SURFACE, fill_opacity=1).move_to([0, 0, 0])
        d_t = T("d", 26, ACCENT).move_to(d)
        student = entity("STUDENT").move_to([-3, -1.6, 0])
        faculty = entity("FACULTY").move_to([3, -1.6, 0])
        cgpa = ellipse_attr("CGPA").move_to([-3, -3.0, 0])
        salary = ellipse_attr("Salary").move_to([3, -3.0, 0])

        self.play(FadeIn(person), FadeIn(pid), FadeIn(pname), Create(connect(person, pid)), Create(connect(person, pname)))
        cap = self.caption("Start with a general entity: PERSON")
        double = VGroup(Line(person.get_bottom() + LEFT * 0.06, d.get_top() + LEFT * 0.06),
                        Line(person.get_bottom() + RIGHT * 0.06, d.get_top() + RIGHT * 0.06)).set_stroke(MUTED, 2.5)
        self.play(Create(double), FadeIn(d), Write(d_t))
        self.play(FadeIn(student), FadeIn(faculty), Create(connect(d, student)), Create(connect(d, faculty)))
        cap = self.caption("Specialization (top-down): PERSON splits into STUDENT and FACULTY", cap)
        self.wait(1.2)
        self.play(FadeIn(cgpa), FadeIn(salary), Create(connect(student, cgpa)), Create(connect(faculty, salary)))
        cap = self.caption("Each sub-type adds its own attributes and inherits ID and Name", cap)
        self.wait(1.2)
        cap = self.caption("'d' = disjoint (a person is one or the other); double line = total participation", cap)
        self.wait(1.5)
        up = Arrow([0, -2.6, 0], [0, -0.5, 0], color=GOOD, stroke_width=6)
        self.play(GrowArrow(up))
        self.caption("Generalization (bottom-up): combine common features into a super-type", cap, GOOD)
        self.outro()


class ERToTables(ThemedScene):
    unit = "1"

    def construct(self):
        self.header("Converting ER to Tables")
        s = entity("STUDENT", 2.0, 0.7).move_to([-3, 2.0, 0])
        e = diamond("ENROLLS", 2.0, 1.0).move_to([0, 2.0, 0])
        c = entity("COURSE", 2.0, 0.7).move_to([3, 2.0, 0])
        er = VGroup(connect(s, e), connect(e, c), s, e, c)
        self.play(FadeIn(er))
        cap = self.caption("Rule 1: every strong entity becomes a table")

        st = grid([["Roll_No", "Name", "Email"], ["101", "Asha", "asha@uni.in"]], [1.4, 1.2, 2.0], 0.5, PRIMARY, 18, key_cols=(0,))
        co = grid([["Course_ID", "Title"], ["DB01", "DBMS"]], [1.6, 1.5], 0.5, PRIMARY, 18, key_cols=(0,))
        en = grid([["Roll_No", "Course_ID", "Grade"], ["101", "DB01", "A"]], [1.4, 1.6, 1.1], 0.5, ACCENT, 18, key_cols=(0, 1))
        st.move_to([-4.0, 0.0, 0])
        co.move_to([4.0, 0.0, 0])
        en.move_to([0, -1.8, 0])
        self.play(TransformFromCopy(s, st), TransformFromCopy(c, co))
        self.wait(1)
        cap = self.caption("Rule 2: an M:N relationship becomes its own table", cap)
        self.play(TransformFromCopy(e, en))
        self.wait(0.8)
        cap = self.caption("Its key = both foreign keys together (Roll_No, Course_ID)", cap)
        fk1 = CurvedArrow(en[0][0].get_left(), st[0][0].get_bottom() + DOWN * 0.5, color=ACCENT, angle=-PI / 3)
        fk2 = CurvedArrow(en[0][1].get_right() + RIGHT * 0.1, co[0][0].get_bottom() + DOWN * 0.5, color=ACCENT, angle=PI / 3)
        self.play(Create(fk1), Create(fk2))
        self.wait(1)
        cap = self.caption("Rule 3 (1:N): put the foreign key in the table on the N side", cap)
        self.wait(1.5)
        self.caption("Rule 4: a multivalued attribute (Phone) gets a separate table", cap)
        self.outro()


# ================================================================== UNIT II
class RelationTerms(ThemedScene):
    unit = "2"

    def construct(self):
        self.header("The Relational Model", "Vocabulary you must know")
        rows = [["Roll_No", "Name", "Dept", "CGPA"],
                ["101", "Asha", "ECE", "8.9"], ["102", "Ravi", "CS", "7.6"],
                ["103", "Meera", "ECE", "9.1"], ["104", "John", "IT", "6.8"]]
        table = grid(rows, 2.0, 0.6, UNIT_COLOR["2"], 24, key_cols=(0,)).shift(DOWN * 0.3)
        self.play(FadeIn(table))
        cap = self.caption("Relation = a table")
        self.wait(1)
        row = SurroundingRectangle(table[2], color=ACCENT, buff=0.04)
        self.play(Create(row))
        cap = self.caption("Tuple = one row (one student)", cap)
        self.wait(1)
        col = SurroundingRectangle(column(table, 3), color=GOOD, buff=0.04)
        self.play(ReplacementTransform(row, col))
        cap = self.caption("Attribute = one column.  Domain = its allowed values (CGPA: 0.0 - 10.0)", cap)
        self.wait(1.5)
        key = SurroundingRectangle(column(table, 0), color=ACCENT, buff=0.04)
        self.play(ReplacementTransform(col, key))
        cap = self.caption("Primary key: unique and never NULL (entity integrity)", cap)
        self.wait(1.5)
        self.play(FadeOut(key))
        cap = self.caption("Degree = 4 columns.  Cardinality = 4 rows", cap)
        self.outro()


class RelationalAlgebra(ThemedScene):
    unit = "2"

    def construct(self):
        self.header("Relational Algebra", "σ select  ·  π project  ·  ⋈ join")
        rows = [["Roll", "Name", "Dept", "CGPA"], ["1", "Asha", "ECE", "8.9"], ["2", "Ravi", "CS", "7.6"],
                ["3", "Meera", "ECE", "9.1"], ["4", "John", "IT", "6.8"]]
        table = grid(rows, 1.6, 0.55, UNIT_COLOR["2"], 22).move_to([-2.8, -0.6, 0])
        self.play(FadeIn(table))

        op = mono("σ Dept='ECE' (STUDENT)", 28, ACCENT).move_to([3.4, 1.0, 0])
        self.play(Write(op))
        cap = self.caption("SELECT (σ) keeps only the rows that match the condition")
        keep, drop = VGroup(table[1], table[3]), VGroup(table[2], table[4])
        self.play(*[r[i][0].animate.set_fill(GOOD, 0.35) for r in keep for i in range(4)])
        self.play(drop.animate.set_opacity(0.15))
        sql = mono("SQL: SELECT * FROM student WHERE dept='ECE';", 20, MUTED).next_to(op, DOWN, buff=0.4)
        self.play(FadeIn(sql))
        self.wait(1.5)

        op2 = mono("π Name, CGPA (STUDENT)", 28, ACCENT).move_to(op)
        self.play(table.animate.set_opacity(1), *[r[i][0].animate.set_fill(SURFACE, 1) for r in keep for i in range(4)],
                  Transform(op, op2), FadeOut(sql))
        cap = self.caption("PROJECT (π) keeps only the columns you ask for", cap)
        self.play(column(table, 0).animate.set_opacity(0.15), column(table, 2).animate.set_opacity(0.15))
        sql2 = mono("SQL: SELECT name, cgpa FROM student;", 20, MUTED).next_to(op, DOWN, buff=0.4)
        self.play(FadeIn(sql2))
        self.wait(1.5)
        self.play(FadeOut(table), FadeOut(op), FadeOut(sql2))

        s = grid([["Roll", "Dept"], ["1", "ECE"], ["2", "CS"]], 1.3, 0.5, UNIT_COLOR["2"], 20).move_to([-5, 0, 0])
        d = grid([["Dept", "HOD"], ["ECE", "Dr. Rao"], ["CS", "Dr. Iyer"]], 1.4, 0.5, UNIT_COLOR["2"], 20).move_to([-1.6, 0, 0])
        j = T("⋈", 60, ACCENT).move_to([-3.25, 0, 0])
        res = grid([["Roll", "Dept", "HOD"], ["1", "ECE", "Dr. Rao"], ["2", "CS", "Dr. Iyer"]], 1.4, 0.5, GOOD, 20).move_to([3.6, 0, 0])
        cap = self.caption("NATURAL JOIN (⋈) glues rows that share the same Dept", cap)
        self.play(FadeIn(s), FadeIn(d), Write(j))
        self.play(Indicate(column(s, 1), color=ACCENT), Indicate(column(d, 0), color=ACCENT))
        self.play(TransformFromCopy(VGroup(s, d), res))
        self.outro()


class SetOperations(ThemedScene):
    unit = "2"

    def construct(self):
        self.header("Set Operations", "Tables must be union-compatible")

        def venn(kind, label, sql):
            a = Circle(1.0, color=PRIMARY).shift(LEFT * 0.55)
            b = Circle(1.0, color=ACCENT).shift(RIGHT * 0.55)
            shade = {"union": Union(a, b), "inter": Intersection(a, b), "diff": Difference(a, b)}[kind]
            shade.set_fill(GOOD, 0.6).set_stroke(width=0)
            labels = VGroup(T("A", 24, PRIMARY).next_to(a, UP, buff=0.05).shift(LEFT * 0.3),
                            T("B", 24, ACCENT).next_to(b, UP, buff=0.05).shift(RIGHT * 0.3))
            text = VGroup(T(label, 30, TXT, bold=True), mono(sql, 20, MUTED)).arrange(DOWN, buff=0.15)
            text.next_to(VGroup(a, b), DOWN, buff=0.3)
            return VGroup(shade, a, b, labels, text)

        groups = VGroup(venn("union", "A ∪ B", "UNION"), venn("inter", "A ∩ B", "INTERSECT"),
                        venn("diff", "A − B", "EXCEPT / MINUS")).arrange(RIGHT, buff=0.9).shift(DOWN * 0.2)
        captions = ["Union: rows in A or B (duplicates removed)", "Intersection: rows in both A and B",
                    "Difference: rows in A that are not in B"]
        cap = None
        for g, text in zip(groups, captions):
            self.play(FadeIn(g, shift=UP * 0.2))
            cap = self.caption(text, cap)
            self.wait(1.5)
        self.caption("Union-compatible = same number of columns with matching types", cap, ACCENT)
        self.outro()


class FunctionalDependency(ThemedScene):
    unit = "2"

    def construct(self):
        self.header("Functional Dependencies")
        fd = mono("Roll_No  ->  Name", 36, ACCENT).move_to([0, 1.8, 0])
        self.play(Write(fd))
        cap = self.caption("X -> Y : whenever two rows have the same X, they must have the same Y")
        rows = [["Roll_No", "Name", "Course"], ["101", "Asha", "DBMS"], ["101", "Asha", "DSP"], ["102", "Ravi", "DBMS"]]
        t = grid(rows, 1.9, 0.55, UNIT_COLOR["2"], 22).move_to([0, -0.2, 0])
        self.play(FadeIn(t))
        self.play(*[Indicate(t[r][c], color=GOOD) for r in (1, 2) for c in (0, 1)])
        ok = T("Holds ✓", 30, GOOD).next_to(t, RIGHT, buff=0.5)
        self.play(FadeIn(ok))
        self.wait(1.2)
        self.play(FadeOut(t), FadeOut(ok), FadeOut(fd))

        cap = self.caption("Armstrong's axioms let us derive new dependencies", cap)
        axioms = VGroup(
            mono("Reflexivity:   if Y ⊆ X  then  X -> Y", 26),
            mono("Augmentation:  if X -> Y then  XZ -> YZ", 26),
            mono("Transitivity:  if X -> Y and Y -> Z then X -> Z", 26),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).move_to([0, 0.4, 0])
        self.play(LaggedStart(*[FadeIn(a, shift=RIGHT * 0.3) for a in axioms], lag_ratio=0.5))
        self.wait(2)
        self.play(FadeOut(axioms))

        cap = self.caption("Closure: given A->B and B->C, find everything A determines", cap)
        steps = VGroup(mono("{A}+ = { A }", 30), mono("{A}+ = { A, B }        (A -> B)", 30),
                       mono("{A}+ = { A, B, C }     (B -> C)", 30, GOOD)).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        for s in steps:
            self.play(FadeIn(s, shift=RIGHT * 0.2))
            self.wait(0.6)
        self.caption("If X+ contains every attribute, X is a (super) key", cap, GOOD)
        self.outro()


class Normalization(ThemedScene):
    unit = "2"

    def construct(self):
        self.header("Normalization", "One messy table -> clean tables")
        UNIF = UNIT_COLOR["2"]
        stage = T("Unnormalised", 30, BAD, bold=True).to_corner(UR, buff=0.6)
        unf = grid([["OrderID", "Customer", "City", "Items"],
                    ["101", "Asha", "Pune", "Pen, Book"], ["102", "Ravi", "Mumbai", "Pen"]],
                   [1.6, 1.7, 1.6, 2.2], 0.6, BAD, 22)
        self.play(FadeIn(stage), FadeIn(unf))
        cap = self.caption("'Pen, Book' is two values in one cell: not atomic")
        self.play(Indicate(unf[1][3], color=BAD, scale_factor=1.15))
        self.wait(1)

        nf1 = grid([["OrderID", "Item", "Customer", "City", "Price"],
                    ["101", "Pen", "Asha", "Pune", "10"], ["101", "Book", "Asha", "Pune", "150"],
                    ["102", "Pen", "Ravi", "Mumbai", "10"]],
                   [1.5, 1.3, 1.6, 1.6, 1.2], 0.55, UNIF, 22, key_cols=(0, 1))
        s1 = T("1NF", 30, UNIF, bold=True).move_to(stage)
        self.play(ReplacementTransform(unf, nf1), Transform(stage, s1))
        cap = self.caption("1NF: one value per cell. Key = (OrderID, Item)", cap)
        self.wait(1.5)
        cap = self.caption("Problem: Customer depends on OrderID only, Price on Item only (partial)", cap)
        self.play(Indicate(column(nf1, 2), color=ACCENT), Indicate(column(nf1, 4), color=ACCENT))
        self.wait(1)

        orders = grid([["OrderID", "Customer", "City"], ["101", "Asha", "Pune"], ["102", "Ravi", "Mumbai"]],
                      [1.5, 1.6, 1.5], 0.55, UNIF, 20, key_cols=(0,))
        items = grid([["Item", "Price"], ["Pen", "10"], ["Book", "150"]], [1.2, 1.1], 0.55, UNIF, 20, key_cols=(0,))
        lines = grid([["OrderID", "Item"], ["101", "Pen"], ["101", "Book"], ["102", "Pen"]], [1.5, 1.2], 0.55, UNIF, 20,
                     key_cols=(0, 1))
        nf2 = VGroup(orders, lines, items).arrange(RIGHT, buff=0.6, aligned_edge=UP).shift(DOWN * 0.2)
        s2 = T("2NF", 30, ACCENT, bold=True).move_to(stage)
        self.play(ReplacementTransform(nf1, nf2), Transform(stage, s2))
        cap = self.caption("2NF: split so every column depends on the whole key", cap)
        self.wait(1.5)
        cap = self.caption("Problem: OrderID -> Customer -> City (transitive dependency)", cap)
        self.play(Indicate(column(orders, 2), color=ACCENT))
        self.wait(1)

        orders3 = grid([["OrderID", "Customer"], ["101", "Asha"], ["102", "Ravi"]], [1.5, 1.6], 0.55, GOOD, 20, key_cols=(0,))
        cust = grid([["Customer", "City"], ["Asha", "Pune"], ["Ravi", "Mumbai"]], [1.6, 1.5], 0.55, GOOD, 20, key_cols=(0,))
        lines3 = lines.copy()
        items3 = items.copy()
        nf3 = VGroup(orders3, cust, lines3, items3).arrange(RIGHT, buff=0.45, aligned_edge=UP).shift(DOWN * 0.2)
        s3 = T("3NF ✓", 30, GOOD, bold=True).move_to(stage)
        self.play(ReplacementTransform(nf2, nf3), Transform(stage, s3))
        self.caption("3NF: City moves to its own table. No more update anomalies!", cap, GOOD)
        self.outro()


# ================================================================== UNIT III
class SQLCommandFamilies(ThemedScene):
    unit = "3"

    def construct(self):
        self.header("SQL Command Families")
        fams = [("DDL", "Define structure", ["CREATE", "ALTER", "DROP", "TRUNCATE"], PRIMARY),
                ("DML", "Change data", ["SELECT", "INSERT", "UPDATE", "DELETE"], GOOD),
                ("DCL", "Control access", ["GRANT", "REVOKE"], ACCENT),
                ("TCL", "Control transactions", ["COMMIT", "ROLLBACK", "SAVEPOINT"], BAD)]
        cols = VGroup()
        for name, desc, cmds, colour in fams:
            head = box(name, 2.8, 0.9, colour, 34)
            d = T(desc, 22, MUTED)
            cs = VGroup(*[mono(c, 24, colour) for c in cmds]).arrange(DOWN, buff=0.25)
            cols.add(VGroup(head, d, cs).arrange(DOWN, buff=0.35))
        cols.arrange(RIGHT, buff=0.45, aligned_edge=UP).shift(DOWN * 0.3)
        cap = None
        for col, (name, desc, _, _) in zip(cols, fams):
            self.play(FadeIn(col[0], shift=DOWN * 0.2), FadeIn(col[1]))
            self.play(LaggedStart(*[FadeIn(c, shift=RIGHT * 0.2) for c in col[2]], lag_ratio=0.2))
            cap = self.caption(f"{name}: {desc.lower()}", cap)
            self.wait(0.8)
        self.outro()


class SQLQueryOrder(ThemedScene):
    unit = "3"

    def construct(self):
        self.header("How SQL Really Runs", "Written order vs execution order")
        written = ["SELECT dept, AVG(salary)", "FROM employee", "WHERE salary > 20000", "GROUP BY dept",
                   "HAVING AVG(salary) > 50000", "ORDER BY 2 DESC", "LIMIT 3;"]
        lines = VGroup(*[mono(s, 24) for s in written]).arrange(DOWN, aligned_edge=LEFT, buff=0.28)
        lines.move_to([-3.4, -0.4, 0])
        w_title = T("You write", 26, MUTED).next_to(lines, UP, buff=0.4)
        self.play(FadeIn(w_title), LaggedStart(*[FadeIn(l) for l in lines], lag_ratio=0.15))
        cap = self.caption("You write SELECT first, but the database does it almost last")
        self.wait(1)

        order = [1, 2, 3, 4, 0, 5, 6]  # FROM, WHERE, GROUP BY, HAVING, SELECT, ORDER BY, LIMIT
        e_title = T("It executes", 26, UNIT_COLOR["3"]).move_to([3.6, w_title.get_y(), 0])
        self.play(FadeIn(e_title))
        targets = VGroup()
        for n, idx in enumerate(order):
            num = T(f"{n + 1}", 24, UNIT_COLOR["3"], bold=True)
            word = mono(written[idx].split(" ")[0] + (" BY" if " BY" in written[idx] else ""), 26, TXT)
            targets.add(VGroup(num, word).arrange(RIGHT, buff=0.3))
        targets.arrange(DOWN, aligned_edge=LEFT, buff=0.3).move_to([3.6, -0.4, 0])
        notes = ["pick the table", "filter rows", "make groups", "filter groups", "compute columns", "sort", "cut to 3"]
        for n, idx in enumerate(order):
            self.play(Indicate(lines[idx], color=UNIT_COLOR["3"]), TransformFromCopy(lines[idx], targets[n]), run_time=0.9)
            cap = self.caption(f"Step {n + 1}: {notes[n]}", cap)
        self.wait(1)
        self.caption("That's why you can't use a SELECT alias inside WHERE", cap, ACCENT)
        self.outro()


class SQLJoins(ThemedScene):
    unit = "3"

    def construct(self):
        self.header("SQL Joins", "Customers  ⟷  Orders")

        def venn(kind, label):
            a = Circle(0.9, color=PRIMARY).shift(LEFT * 0.5)
            b = Circle(0.9, color=ACCENT).shift(RIGHT * 0.5)
            shade = {"inner": Intersection(a, b), "left": a.copy(), "right": b.copy(), "full": Union(a, b)}[kind]
            shade.set_fill(GOOD, 0.6).set_stroke(width=0)
            t = mono(label, 22).next_to(VGroup(a, b), DOWN, buff=0.3)
            return VGroup(shade, a, b, t)

        vs = VGroup(venn("inner", "INNER JOIN"), venn("left", "LEFT JOIN"), venn("right", "RIGHT JOIN"),
                    venn("full", "FULL JOIN")).arrange(RIGHT, buff=0.6).shift(UP * 0.4)
        legend = VGroup(T("Customers", 22, PRIMARY), T("Orders", 22, ACCENT)).arrange(RIGHT, buff=1).next_to(vs, DOWN, buff=0.5)
        self.play(FadeIn(legend))
        texts = ["Only customers who placed orders", "All customers, with orders where they exist (else NULL)",
                 "All orders, with customer details where they exist", "Everyone from both sides (MySQL: LEFT UNION RIGHT)"]
        cap = None
        for v, t in zip(vs, texts):
            self.play(FadeIn(v, shift=UP * 0.2))
            cap = self.caption(t, cap)
            self.wait(1.4)
        self.outro()


class TriggerFlow(ThemedScene):
    unit = "3"

    def construct(self):
        self.header("Triggers", "Code that runs automatically")
        stmt = mono("INSERT INTO orders VALUES (7, 'Pen', 3);", 24, ACCENT).move_to([0, 2.0, 0])
        orders = grid([["id", "item", "qty"], ["6", "Book", "1"]], 1.3, 0.55, UNIT_COLOR["3"], 22).move_to([-4, 0, 0])
        inv = grid([["item", "stock"], ["Pen", "50"], ["Book", "20"]], 1.5, 0.55, UNIT_COLOR["3"], 22).move_to([4, 0, 0])
        o_lab = T("orders", 22, MUTED).next_to(orders, UP)
        i_lab = T("inventory", 22, MUTED).next_to(inv, UP)
        self.play(FadeIn(orders), FadeIn(inv), FadeIn(o_lab), FadeIn(i_lab))
        self.play(Write(stmt))
        cap = self.caption("1. A normal INSERT arrives")

        new_row = grid([["7", "Pen", "3"]], 1.3, 0.55, UNIT_COLOR["3"], 22)
        for cell in new_row[0]:
            cell[0].set_fill(SURFACE, 1)
            cell[1].set_color(TXT)
        new_row.next_to(orders, DOWN, buff=0)
        self.play(TransformFromCopy(stmt, new_row))
        cap = self.caption("2. The row is added to orders", cap)

        bolt = box("AFTER INSERT trigger fires", 4.2, 0.8, ACCENT, 24).move_to([0, -0.2, 0])
        self.play(FadeIn(bolt, scale=0.8), Flash(bolt, color=ACCENT))
        cap = self.caption("3. The trigger runs by itself, no extra command needed", cap)
        code = mono("UPDATE inventory SET stock = stock - NEW.qty\nWHERE item = NEW.item;", 20, MUTED).next_to(bolt, DOWN, buff=0.4)
        self.play(FadeIn(code))
        self.play(Indicate(inv[1][1], color=ACCENT))
        new_stock = T("47", 22, GOOD).move_to(inv[1][1][1])
        self.play(Transform(inv[1][1][1], new_stock))
        self.caption("4. Inventory stays in sync: 50 - 3 = 47", cap, GOOD)
        self.outro()


# ================================================================== UNIT IV
class ACIDProperties(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("ACID Properties", "Transfer ₹300 from A to B")
        a = box("Account A: 1000", 3.4, 0.8, PRIMARY, 24).move_to([-3, 1.4, 0])
        b = box("Account B: 500", 3.4, 0.8, PRIMARY, 24).move_to([3, 1.4, 0])
        arrow = Arrow(a.get_right(), b.get_left(), color=ACCENT)
        self.play(FadeIn(a), FadeIn(b), GrowArrow(arrow))
        self.play(Transform(a, box("Account A: 700", 3.4, 0.8, PRIMARY, 24).move_to(a)),
                  Transform(b, box("Account B: 800", 3.4, 0.8, PRIMARY, 24).move_to(b)))
        props = [("A", "Atomicity", "Both updates happen, or neither", PRIMARY),
                 ("C", "Consistency", "Total stays 1500 before and after", GOOD),
                 ("I", "Isolation", "Others never see the half-done state", ACCENT),
                 ("D", "Durability", "Once committed, a crash can't undo it", BAD)]
        cards = VGroup()
        for letter, name, desc, colour in props:
            l = T(letter, 64, colour, HEAD_FONT, bold=True)
            n = T(name, 26, TXT, bold=True)
            d = fit(T(desc, 18, MUTED), 2.7)
            content = VGroup(l, n, d).arrange(DOWN, buff=0.2)
            frame = RoundedRectangle(corner_radius=0.2, width=3.0, height=2.6, stroke_color=colour,
                                     fill_color=SURFACE, fill_opacity=1).move_to(content)
            cards.add(VGroup(frame, content))
        cards.arrange(RIGHT, buff=0.25).shift(DOWN * 1.4)
        cap = None
        for card, (_, name, desc, _) in zip(cards, props):
            self.play(FadeIn(card, shift=UP * 0.3))
            cap = self.caption(f"{name}: {desc}", cap)
            self.wait(1.2)
        self.outro()


class TransactionStates(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("Transaction States")
        col = UNIT_COLOR["4"]
        active = box("Active", 2.6, 0.9, col, 26).move_to([-5, 0.8, 0])
        partial = box("Partially\ncommitted", 2.6, 1.1, col, 22).move_to([-0.8, 1.8, 0])
        committed = box("Committed", 2.6, 0.9, GOOD, 26).move_to([3.6, 1.8, 0])
        failed = box("Failed", 2.6, 0.9, BAD, 26).move_to([-0.8, -1.2, 0])
        aborted = box("Aborted", 2.6, 0.9, BAD, 26).move_to([3.6, -1.2, 0])
        terminated = box("Terminated", 2.6, 0.9, MUTED, 24).move_to([5.6, 0.3, 0]).scale(0.9)

        def arr(x, y, label=None):
            a = Arrow(x.get_right() if x.get_x() < y.get_x() - 1 else x.get_bottom(),
                      y.get_left() if x.get_x() < y.get_x() - 1 else y.get_top(), color=MUTED, buff=0.1)
            g = VGroup(a)
            if label:
                g.add(T(label, 18, MUTED).next_to(a.get_center(), UP, buff=0.12))
            return g

        steps = [
            (active, None, "Active: the transaction is running its reads and writes"),
            (partial, arr(active, partial, "last statement"), "Partially committed: done, but not yet safely on disk"),
            (committed, arr(partial, committed, "written to log"), "Committed: changes are permanent"),
            (failed, arr(active, failed, "error"), "Failed: something went wrong (error, crash, deadlock)"),
            (aborted, arr(failed, aborted, "rollback"), "Aborted: all changes undone; restart or kill"),
        ]
        cap = None
        for node, a, text in steps:
            anims = [FadeIn(node, scale=0.9)]
            if a is not None:
                anims.insert(0, GrowArrow(a[0]))
                anims += [FadeIn(m) for m in a[1:]]
            self.play(*anims)
            cap = self.caption(text, cap)
            self.wait(1)
        pf = Arrow(partial.get_bottom(), failed.get_top(), color=MUTED, buff=0.1)
        self.play(GrowArrow(pf))
        self.play(FadeIn(terminated), GrowArrow(Arrow(committed.get_right(), terminated.get_top(), color=MUTED, buff=0.1)),
                  GrowArrow(Arrow(aborted.get_right(), terminated.get_bottom(), color=MUTED, buff=0.1)))
        self.caption("Either way, the transaction finally ends: Terminated", cap)
        self.outro()


class PrecedenceGraph(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("Conflict Serializability", "Draw the precedence graph")
        col = UNIT_COLOR["4"]
        rows = [["Time", "T1", "T2"], ["1", "R(A)", ""], ["2", "", "W(A)"], ["3", "", "R(B)"], ["4", "W(B)", ""]]
        sched = grid(rows, [1.1, 1.6, 1.6], 0.6, col, 24).move_to([-3.5, -0.4, 0])
        self.play(FadeIn(sched))
        cap = self.caption("Conflict = same item, different transactions, at least one write")

        t1 = Circle(0.55, color=PRIMARY, fill_color=SURFACE, fill_opacity=1).move_to([2.4, -0.4, 0])
        t2 = Circle(0.55, color=ACCENT, fill_color=SURFACE, fill_opacity=1).move_to([5.4, -0.4, 0])
        self.play(FadeIn(t1), FadeIn(t2), Write(T("T1", 28).move_to(t1)), Write(T("T2", 28).move_to(t2)))

        self.play(Indicate(sched[1][1], color=ACCENT), Indicate(sched[2][2], color=ACCENT))
        e1 = CurvedArrow(t1.get_top(), t2.get_top(), angle=-PI / 3, color=PRIMARY)
        self.play(Create(e1))
        cap = self.caption("T1 reads A before T2 writes A  ->  edge T1 -> T2", cap)
        self.wait(1.2)
        self.play(Indicate(sched[3][2], color=ACCENT), Indicate(sched[4][1], color=ACCENT))
        e2 = CurvedArrow(t2.get_bottom(), t1.get_bottom(), angle=-PI / 3, color=ACCENT)
        self.play(Create(e2))
        cap = self.caption("T2 reads B before T1 writes B  ->  edge T2 -> T1", cap)
        self.wait(1.2)
        self.play(e1.animate.set_color(BAD), e2.animate.set_color(BAD))
        self.caption("Cycle found: this schedule is NOT conflict-serializable", cap, BAD)
        self.outro()


class TwoPhaseLocking(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("Two-Phase Locking (2PL)")
        col = UNIT_COLOR["4"]
        matrix = grid([["", "S", "X"], ["S", "✓", "✗"], ["X", "✗", "✗"]], 1.0, 0.7, col, 28).move_to([-4.6, -0.2, 0])
        for r, c in [(1, 1)]:
            matrix[r][c][1].set_color(GOOD)
        for r, c in [(1, 2), (2, 1), (2, 2)]:
            matrix[r][c][1].set_color(BAD)
        lbl = T("Lock compatibility", 22, MUTED).next_to(matrix, UP, buff=0.3)
        self.play(FadeIn(matrix), FadeIn(lbl))
        cap = self.caption("Shared (S) to read, Exclusive (X) to write. Only S + S can coexist")
        self.wait(1.5)

        axes = Axes(x_range=[0, 10, 1], y_range=[0, 4, 1], x_length=6.5, y_length=3.2,
                    axis_config={"color": MUTED, "include_ticks": False}).move_to([2.6, -0.4, 0])
        xl = T("time", 20, MUTED).next_to(axes.x_axis, DOWN, buff=0.15)
        yl = T("locks held", 20, MUTED).next_to(axes.y_axis, UP, buff=0.15)
        self.play(Create(axes), FadeIn(xl), FadeIn(yl))
        pts = [(0, 0), (1, 0), (1, 1), (2.5, 1), (2.5, 2), (4, 2), (4, 3), (5, 3), (5, 2), (6.5, 2), (6.5, 1), (8, 1), (8, 0), (9, 0)]
        path = VMobject(color=col, stroke_width=5).set_points_as_corners([axes.c2p(x, y) for x, y in pts])
        self.play(Create(path), run_time=3)
        lp = Dot(axes.c2p(5, 3), color=ACCENT, radius=0.1)
        self.play(FadeIn(lp), Write(T("lock point", 20, ACCENT).next_to(lp, UP, buff=0.12)))
        g = T("growing", 22, GOOD).move_to(axes.c2p(2.5, 3.4))
        s = T("shrinking", 22, BAD).move_to(axes.c2p(7.2, 3.4))
        self.play(FadeIn(g), FadeIn(s))
        cap = self.caption("Growing phase: only acquire.  Shrinking phase: only release", cap)
        self.wait(1.5)
        self.caption("Strict 2PL: hold X-locks until commit, so no cascading rollbacks", cap, ACCENT)
        self.outro()


class DeadlockWaitFor(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("Deadlock", "Two transactions waiting for each other forever")
        t1 = box("T1", 1.6, 0.9, PRIMARY, 30).move_to([-3.2, 1.0, 0])
        t2 = box("T2", 1.6, 0.9, ACCENT, 30).move_to([3.2, 1.0, 0])
        a = cylinder("A", 1.2, 0.9, MUTED).move_to([-3.2, -1.6, 0])
        b = cylinder("B", 1.2, 0.9, MUTED).move_to([3.2, -1.6, 0])
        self.play(FadeIn(t1), FadeIn(t2), FadeIn(a), FadeIn(b))
        h1 = Arrow(a.get_top(), t1.get_bottom(), color=GOOD)
        h2 = Arrow(b.get_top(), t2.get_bottom(), color=GOOD)
        holds = VGroup(T("holds", 20, GOOD).next_to(h1, LEFT), T("holds", 20, GOOD).next_to(h2, RIGHT))
        self.play(GrowArrow(h1), GrowArrow(h2), FadeIn(holds))
        cap = self.caption("T1 locks A.  T2 locks B.")
        self.wait(0.8)
        w1 = DashedLine(t1.get_right(), b.get_left() + UP * 0.2, color=BAD).add_tip()
        w2 = DashedLine(t2.get_left(), a.get_right() + UP * 0.2, color=BAD).add_tip()
        self.play(Create(w1))
        cap = self.caption("T1 now wants B ... but T2 has it", cap)
        self.play(Create(w2))
        cap = self.caption("T2 now wants A ... but T1 has it.  Nobody can move!", cap)
        self.wait(1)

        self.play(*[FadeOut(m) for m in [a, b, h1, h2, w1, w2, holds]])
        self.play(t1.animate.move_to([-2, 0, 0]), t2.animate.move_to([2, 0, 0]))
        e1 = CurvedArrow(t1.get_top(), t2.get_top(), angle=-PI / 3, color=BAD)
        e2 = CurvedArrow(t2.get_bottom(), t1.get_bottom(), angle=-PI / 3, color=BAD)
        self.play(Create(e1), Create(e2))
        cap = self.caption("Wait-for graph: a cycle means deadlock", cap, BAD)
        self.wait(1.2)
        self.play(t2.animate.set_opacity(0.2), FadeOut(e1), FadeOut(e2))
        self.caption("Fix: abort one victim (T2), roll it back, restart it later", cap, GOOD)
        self.outro()


class LogRecovery(ThemedScene):
    unit = "4"

    def construct(self):
        self.header("Log-Based Recovery", "Write-ahead log + checkpoints")
        entries = ["<T1 start>", "<T1, A, 1000, 700>", "<T1 commit>", "<CHECKPOINT>", "<T2 start>",
                   "<T2, B, 500, 800>", "<T3 start>", "<T3, C, 40, 90>", "<T3 commit>"]
        log = VGroup(*[mono(e, 24) for e in entries]).arrange(DOWN, aligned_edge=LEFT, buff=0.18).move_to([-3.6, -0.4, 0])
        log[3].set_color(ACCENT)
        self.play(LaggedStart(*[FadeIn(l, shift=RIGHT * 0.2) for l in log], lag_ratio=0.15))
        cap = self.caption("Every change is written to the log BEFORE it touches the data")
        self.wait(1)
        crash = T("CRASH", 36, BAD, bold=True).next_to(log, DOWN, buff=0.25)
        self.play(FadeIn(crash, scale=1.5), Flash(crash, color=BAD))
        cap = self.caption("After the crash we scan the log back to the last checkpoint", cap)
        self.play(VGroup(*log[:3]).animate.set_opacity(0.25))
        self.wait(1)

        redo = box("REDO: T3  (committed)", 4.6, 0.9, GOOD, 26).move_to([3.4, 0.8, 0])
        undo = box("UNDO: T2  (no commit)", 4.6, 0.9, BAD, 26).move_to([3.4, -0.6, 0])
        skip = box("Skip: T1  (safe before checkpoint)", 4.6, 0.9, MUTED, 22).move_to([3.4, -2.0, 0])
        self.play(Indicate(log[8], color=GOOD), FadeIn(redo, shift=LEFT * 0.3))
        cap = self.caption("T3 committed: REDO it using the new values", cap)
        self.wait(1)
        self.play(Indicate(log[5], color=BAD), FadeIn(undo, shift=LEFT * 0.3))
        cap = self.caption("T2 never committed: UNDO it using the old values", cap)
        self.wait(1)
        self.play(FadeIn(skip, shift=LEFT * 0.3))
        self.caption("The checkpoint saved us from re-checking T1", cap, ACCENT)
        self.outro()


# ================================================================== UNIT V
class NoSQLTypes(ThemedScene):
    unit = "5"

    def construct(self):
        self.header("Four Families of NoSQL")
        col = UNIT_COLOR["5"]
        kv = VGroup(mono('"user:42" -> "Asha"', 20), mono('"cart:42" -> [3, 7]', 20)).arrange(DOWN, aligned_edge=LEFT)
        doc = mono('{ "name": "Asha",\n  "skills": ["SQL", "C"],\n  "dept": { "id": "ECE" } }', 18)
        cf = grid([["row", "name", "city", "score"], ["42", "Asha", "Pune", ""], ["43", "Ravi", "", "88"]], 1.0, 0.4, col, 16)
        n = [Dot(p, radius=0.14, color=col) for p in ([-0.9, 0.4, 0], [0.9, 0.5, 0], [0, -0.6, 0], [1.4, -0.5, 0])]
        gr = VGroup(*[Line(n[i].get_center(), n[j].get_center(), color=MUTED) for i, j in [(0, 1), (0, 2), (1, 2), (1, 3)]], *n)
        data = [("Key-Value", "Redis, DynamoDB", "sessions, caches", kv),
                ("Document", "MongoDB, CouchDB", "catalogues, profiles", doc),
                ("Column-Family", "Cassandra, HBase", "IoT, logs at scale", cf),
                ("Graph", "Neo4j", "social networks, fraud", gr)]
        cards = VGroup()
        for name, ex, use, visual in data:
            frame = RoundedRectangle(corner_radius=0.2, width=6.2, height=2.7, stroke_color=col, fill_color=SURFACE, fill_opacity=1)
            title = T(name, 28, col, bold=True)
            sub = T(f"{ex}  ·  {use}", 18, MUTED)
            fit(visual, 5.0)
            if visual.height > 1.3:
                visual.scale_to_fit_height(1.3)
            content = VGroup(title, visual, sub).arrange(DOWN, buff=0.2).move_to(frame)
            cards.add(VGroup(frame, content))
        cards.arrange_in_grid(2, 2, buff=0.3).shift(DOWN * 0.45)
        for card in cards:
            self.play(FadeIn(card, shift=UP * 0.2))
            self.wait(1.4)
        self.wait(1)
        self.outro()


class CAPTheorem(ThemedScene):
    unit = "5"

    def construct(self):
        self.header("CAP Theorem")
        col = UNIT_COLOR["5"]
        pts = {"C": [0, 1.7, 0], "A": [-3.2, -2.0, 0], "P": [3.2, -2.0, 0]}
        tri = Polygon(*pts.values(), stroke_color=MUTED, stroke_width=3)
        nodes = {}
        for k, p in pts.items():
            c = Circle(0.5, color=col, fill_color=SURFACE, fill_opacity=1).move_to(p)
            nodes[k] = VGroup(c, T(k, 34, col, bold=True).move_to(c))
        names = {"C": "Consistency: every read sees the latest write",
                 "A": "Availability: every request gets an answer",
                 "P": "Partition tolerance: keeps working if the network splits"}
        self.play(Create(tri))
        cap = None
        for k in "CAP":
            self.play(FadeIn(nodes[k], scale=0.8))
            cap = self.caption(names[k], cap)
            self.wait(1)
        cap = self.caption("When the network splits, you can only keep C or A, not both", cap, ACCENT)
        self.wait(1)
        edges = [("CP", [2.6, 0.1, 0], "MongoDB, HBase"), ("AP", [-2.6, 0.1, 0], "Cassandra, CouchDB"),
                 ("CA", [0, -2.6, 0], "Single-server RDBMS (no partition)")]
        for lab, pos, ex in edges:
            g = VGroup(T(lab, 26, TXT, bold=True), T(ex, 18, MUTED)).arrange(DOWN, buff=0.08).move_to(pos)
            self.play(FadeIn(g))
            self.wait(0.6)
        self.caption("Relational = ACID.  Many NoSQL systems = BASE (eventually consistent)", cap)
        self.outro()
